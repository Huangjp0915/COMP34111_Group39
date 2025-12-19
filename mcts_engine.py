"""
MCTS引擎 - PUCT版本 (MoHex Gamma Integration) + RAVE/AMAF
EH-v7.1

核心特性：
1. Gamma to Rho Normalization:
   在 _expand 中计算所有动作的 Gamma 值，并归一化为概率分布 (Prior)。
2. Context Awareness:
   正确传递 opp_move 给 PatternRecognizer，激活局部应手 (Local Patterns)。
3. Dependency Injection:
   接收外部注入的 Evaluator 和 Recognizer。
4. RAVE/AMAF:
   在 Select 阶段将 Q 与 AMAF(Q_rave) 混合；在 Backprop 阶段更新 AMAF 统计。
   可选浅 Rollout（rollout_depth>0）增强 AMAF 信号；rollout_depth=0 也可工作（仅用树路径动作更新）。
"""

import copy
import math
import time
import random
from typing import Optional, List, Tuple, Dict

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# 占位导入，实际运行时会被 SmartHexAgent 注入的实例覆盖
from .connectivity_evaluator import ConnectivityEvaluator
from .pattern_recognizer import PatternRecognizer
from .utils import hash_board


class MCTSNode:
    """
    MCTS 节点
    """
    __slots__ = (
        'board', 'move', 'parent', 'children',
        'visits', 'wins', 'player', 'prior', 'untried_moves',
        'rave_stats'
    )

    def __init__(
        self,
        board: Board,
        move: Optional[Move] = None,
        parent: Optional["MCTSNode"] = None,
        player: Optional[Colour] = None,
        prior: float = 0.0
    ):
        self.board = board
        self.move = move          # 到达此节点的动作 (即 Parent 做的动作)
        self.parent = parent
        self.children: List["MCTSNode"] = []

        self.visits = 0
        self.wins = 0.0           # 累计价值 (相对于上一手行动者，即 self.move 的发起者)
        self.player = player      # 当前节点执子方 (Next to move)
        self.prior = prior        # 先验概率 (从 Gamma 归一化而来)

        # 未尝试动作列表: 存储 (Move, Prior) 元组，作为栈使用
        # Lazy generation: 首次 expand 时才生成
        self.untried_moves: Optional[List[Tuple[Move, float]]] = None

        # RAVE/AMAF 统计：key=(x,y) -> [rave_visits:int, rave_wins:float]
        self.rave_stats: Dict[Tuple[int, int], List[float]] = {}

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        # 简单检查：实际胜负通常在 Simulate 阶段由 Evaluator 判断
        return False


class MCTSEngine:
    """
    MCTS 搜索引擎 (PUCT + Gamma Prior + RAVE)
    """
    def __init__(self, colour: Colour):
        self.colour = colour

        # [依赖注入] 这里的实例会被 SmartHexAgent 覆盖
        self.connectivity_evaluator = ConnectivityEvaluator()
        self.pattern_recognizer = PatternRecognizer(colour)

        self.root_node: Optional[MCTSNode] = None
        self.last_move: Optional[Move] = None      # 真正的对手上一手 (用于 Root 扩展)
        self.last_board_hash: Optional[str] = None

        # PUCT 参数
        self.c_puct_base = 19652
        self.c_puct_init = 1.25

        # RAVE 参数
        self.use_rave = True
        self.rave_equiv = 300.0     # 越大：RAVE 影响更久；越小：更快回到真实 Q
        self.rollout_depth = 0      # 0=不做 rollout（仅用树路径动作更新AMAF）；>0 可增强 AMAF 信号

    def reset_tree(self):
        """强制重置搜索树 (用于 Swap 或纠错)"""
        self.root_node = None
        self.last_move = None
        self.last_board_hash = None

    def search(self, board: Board, time_budget: float) -> Move:
        """
        执行 MCTS 搜索
        """
        # 1. 根节点管理 (复用 vs 重建)
        current_board_hash = hash_board(board)

        if self.root_node and self.last_board_hash:
            # 尝试复用子树
            found_child = False
            # 只有当哈希不匹配 (说明棋盘变了) 时才去搜寻子节点
            if self.last_board_hash != current_board_hash:
                for child in self.root_node.children:
                    # 匹配逻辑：
                    # 1. 对手确实走了这步 (last_move)
                    # 2. 棋盘状态哈希一致
                    if self.last_move and child.move == self.last_move:
                        if hash_board(child.board) == current_board_hash:
                            self.root_node = child
                            self.root_node.parent = None  # 断开父链接，成为新根
                            found_child = True
                            break

                if not found_child:
                    # 没找到对应子树 (可能是对手走了我们未扩展的点)，重置
                    self.root_node = MCTSNode(board, player=self.colour)
        else:
            self.root_node = MCTSNode(board, player=self.colour)

        self.last_board_hash = current_board_hash

        # 2. 搜索主循环
        start_time = time.perf_counter()
        iterations = 0

        while True:
            if iterations & 255 == 0:  # 每 256 次检查一次时间
                elapsed = time.perf_counter() - start_time
                if elapsed >= time_budget:
                    if iterations > 10:
                        break

            self._mcts_iteration(self.root_node)
            iterations += 1

        # 3. 选择最佳移动
        best_move = self._select_best_move(self.root_node)

        if best_move is None:
            return self._fallback(board)
        return best_move

    # ---------------- RAVE helpers ----------------

    def _rave_beta(self, n_child: int, n_rave: int) -> float:
        """
        RAVE 混合系数 beta（常见 MoGo/MoHex 风格）：
        beta = N_rave / (N + N_rave + 4*N*N_rave / k)
        """
        if n_rave <= 0:
            return 0.0
        n = float(max(n_child, 0))
        r = float(max(n_rave, 0))
        k = self.rave_equiv + 1e-9
        return r / (n + r + 4.0 * n * r / k + 1e-9)

    # ---------------- one iteration ----------------

    def _mcts_iteration(self, root: MCTSNode) -> None:
        """一次 MCTS 迭代：Select -> Expand -> Simulate -> Backprop (含 RAVE)"""
        node, path = self._select_with_path(root)

        # Expand
        if not node.is_terminal():
            if not node.is_fully_expanded():
                node = self._expand(node)
                path.append(node)

        # 构造树路径动作序列（按时间顺序）
        tree_moves: List[Tuple[Move, Colour]] = []
        for i in range(len(path) - 1):
            mv = path[i + 1].move
            if mv is not None:
                # path[i] 执行 mv
                tree_moves.append((mv, path[i].player))

        # Simulate / Evaluate
        score, rollout_moves = self._simulate(node)

        # 整条“模拟中发生过的动作序列”
        sim_moves = tree_moves + rollout_moves

        # 为每个 path 节点记录：从它之后开始的 sim_moves 下标
        # depth i 的节点之后发生的树动作从 tree_moves[i] 开始
        pos_map: Dict[MCTSNode, int] = {}
        for i, nd in enumerate(path):
            if i < len(path) - 1:
                pos_map[nd] = i
            else:
                pos_map[nd] = len(tree_moves)  # leaf 之后只剩 rollout

        # Backprop (含 AMAF)
        self._backpropagate_with_rave(node, score, sim_moves, pos_map)

    # ---------------- selection (PUCT + RAVE) ----------------

    def _select_with_path(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """PUCT 选择策略（Q 与 AMAF(Q_rave) 混合），同时返回路径"""
        current = node
        path = [current]

        while current.children:
            if not current.is_fully_expanded():
                return current, path

            best_child = None
            best_score = -float('inf')
            sqrt_parent_visits = math.sqrt(current.visits + 1e-8)
            c_puct = self.c_puct_init + math.log((current.visits + self.c_puct_base + 1) / self.c_puct_base)

            for child in current.children:
                # Q
                if child.visits > 0:
                    q_val = child.wins / child.visits
                else:
                    q_val = 0.0  # FPU

                # RAVE blend：使用 current 节点的 AMAF 统计来评估 child.move
                if self.use_rave and child.move is not None:
                    key = (child.move.x, child.move.y)
                    stats = current.rave_stats.get(key)
                    if stats is not None:
                        rave_n = int(stats[0])
                        if rave_n > 0:
                            q_rave = float(stats[1]) / rave_n
                            beta = self._rave_beta(child.visits, rave_n)
                            q_val = (1.0 - beta) * q_val + beta * q_rave

                # U (PUCT)
                u_val = c_puct * child.prior * sqrt_parent_visits / (1 + child.visits)
                score = q_val + u_val

                if score > best_score:
                    best_score = score
                    best_child = child

            current = best_child  # type: ignore
            path.append(current)

        return current, path

    # ---------------- expansion ----------------

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        扩展节点：
        1) (首次) 生成所有合法动作，计算 Gamma -> Prior，存入 untried_moves。
        2) (每次) 从 untried_moves 中弹出一个 Prior 最高的动作，创建子节点。
        """
        if node.untried_moves is None:
            size = node.board.size
            valid_moves: List[Move] = []

            for x in range(size):
                for y in range(size):
                    if node.board.tiles[x][y].colour is None:
                        valid_moves.append(Move(x, y))

            if not valid_moves:
                node.untried_moves = []
                return node

            # 确定对手上一手（用于 Local Pattern）
            opp_move_ctx = node.move
            if node == self.root_node:
                opp_move_ctx = self.last_move

            # 批量计算 Gamma
            gammas = self.pattern_recognizer.get_all_gammas(node.board, valid_moves, opp_move_ctx)

            # 归一化 Gamma -> Prior (Rho)
            total_gamma = sum(gammas.values()) + 1e-9

            weighted_moves: List[Tuple[Move, float]] = []
            for mv in valid_moves:
                g = gammas.get((mv.x, mv.y), 1.0)
                prior = g / total_gamma
                weighted_moves.append((mv, prior))

            # pop() 取尾部，所以从小到大排序
            weighted_moves.sort(key=lambda x: x[1])
            node.untried_moves = weighted_moves

        if not node.untried_moves:
            return node

        move, prior = node.untried_moves.pop()

        new_board = copy.deepcopy(node.board)
        new_board.set_tile_colour(move.x, move.y, node.player)

        next_player = Colour.opposite(node.player)

        child = MCTSNode(
            board=new_board,
            move=move,
            parent=node,
            player=next_player,
            prior=prior
        )

        node.children.append(child)
        return child

    # ---------------- simulation / evaluation ----------------

    def _simulate(self, node: MCTSNode) -> Tuple[float, List[Tuple[Move, Colour]]]:
        """
        返回:
          score: 相对于 node.player 的优势分数（[-1,1]）
          rollout_moves: [(move, who_played), ...] 从 node 往后 rollout 的动作序列
        """
        if self.rollout_depth <= 0:
            score = self.connectivity_evaluator.evaluate_leaf(node.board, node.player)
            return score, []

        sim_board = copy.deepcopy(node.board)
        player = node.player
        rollout_seq: List[Tuple[Move, Colour]] = []

        for _ in range(self.rollout_depth):
            size = sim_board.size
            valid: List[Move] = []
            for x in range(size):
                for y in range(size):
                    if sim_board.tiles[x][y].colour is None:
                        valid.append(Move(x, y))
            if not valid:
                break

            mv = random.choice(valid)  # 轻量版：随机；需要更强可改成 Gamma 加权采样
            sim_board.set_tile_colour(mv.x, mv.y, player)
            rollout_seq.append((mv, player))
            player = Colour.opposite(player)

        score = self.connectivity_evaluator.evaluate_leaf(sim_board, node.player)
        return score, rollout_seq

    # ---------------- backprop (standard + RAVE/AMAF) ----------------

    def _backpropagate_with_rave(
        self,
        node: MCTSNode,
        score: float,
        sim_moves: List[Tuple[Move, Colour]],
        pos_map: Dict[MCTSNode, int]
    ) -> None:
        """
        回传（含 AMAF）：
        - 常规：更新 visits/wins
        - AMAF：对每个祖先节点，把“该节点之后发生过、且由该节点 player 下的动作”记入 rave_stats
        """
        current = node
        current_score = score

        while current:
            current.visits += 1
            current.wins += current_score

            if self.use_rave and sim_moves:
                start = pos_map.get(current, 0)
                # 对于 current 节点：AMAF 更新只统计 current.player 执行过的动作
                for mv, who in sim_moves[start:]:
                    if who != current.player:
                        continue
                    # 必须在 current 局面中仍为可下点才算 AMAF（否则是非法动作）
                    if current.board.tiles[mv.x][mv.y].colour is not None:
                        continue

                    key = (mv.x, mv.y)
                    stats = current.rave_stats.get(key)
                    if stats is None:
                        # visits=1, wins=current_score
                        current.rave_stats[key] = [1.0, float(current_score)]
                    else:
                        stats[0] += 1.0
                        stats[1] += float(current_score)

            # 视角翻转（零和）
            current_score = -current_score
            current = current.parent

    # ---------------- final move choice ----------------

    def _select_best_move(self, root: MCTSNode) -> Optional[Move]:
        """选择访问次数最多的子节点 (Robust Child)"""
        if not root.children:
            return None
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _fallback(self, board: Board) -> Move:
        """兜底：返回第一个合法点"""
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    return Move(x, y)
        return Move(-1, -1)
