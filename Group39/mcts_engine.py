"""
MCTS引擎 - PUCT版本 (MoHex Gamma Integration)
EH-v7.0

核心升级：
1. Gamma to Rho Normalization: 
   在 _expand 中计算所有动作的 Gamma 值，并归一化为概率分布 (Prior)。
2. Context Awareness:
   正确传递 opp_move 给 PatternRecognizer，激活局部应手 (Local Patterns)。
3. Dependency Injection:
   接收外部注入的 Evaluator 和 Recognizer。
"""

import copy
import math
import time
import random
from typing import Optional, List, Tuple

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
    __slots__ = ('board', 'move', 'parent', 'children', 'visits', 'wins', 'player', 'prior', 'untried_moves')

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
        self.children: List[MCTSNode] = []

        self.visits = 0
        self.wins = 0.0           # 累计价值 (相对于上一手行动者，即 self.move 的发起者)
        self.player = player      # 当前节点执子方 (Next to move)
        self.prior = prior        # 先验概率 (从 Gamma 归一化而来)

        # 未尝试动作列表: 存储 (Move, Prior) 元组，作为栈使用
        # Lazy generation: 首次 expand 时才生成
        self.untried_moves: Optional[List[Tuple[Move, float]]] = None

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        # 简单检查：是否还有空位
        # 实际胜负通常在 Simulate 阶段由 Evaluator 判断，或者 Board.has_ended
        # 这里为了效率，主要依赖 untried_moves 是否为空来判断扩展性
        return False 

class MCTSEngine:
    """
    MCTS 搜索引擎 (PUCT + Gamma Prior)
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
                            self.root_node.parent = None # 断开父链接，成为新根
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
            # 时间控制
            if iterations & 255 == 0: # 每 256 次检查一次时间，减少系统调用开销
                elapsed = time.perf_counter() - start_time
                if elapsed >= time_budget:
                    # 保证至少有一定量的搜索
                    if iterations > 10: 
                        break
            
            self._mcts_iteration(self.root_node)
            iterations += 1

        # 3. 选择最佳移动
        best_move = self._select_best_move(self.root_node)
        
        # 兜底
        if best_move is None:
            return self._fallback(board)

        return best_move

    def _mcts_iteration(self, root: MCTSNode) -> None:
        """一次 MCTS 迭代：Select -> Expand -> Simulate -> Backprop"""
        node = self._select(root)
        
        # 如果不是终局且未完全扩展，则扩展一步
        # 注意：is_terminal 这里是简单的占位，主要靠 untried_moves 判断
        if not node.is_terminal():
            if not node.is_fully_expanded():
                node = self._expand(node)
            else:
                # 已完全扩展的非终局节点 (极为罕见，通常意味着选到了叶子但没扩展空间?)
                pass
        
        # 模拟/评估
        value = self._simulate(node)
        
        # 回传
        self._backpropagate(node, value)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """PUCT 选择策略"""
        current = node
        
        # 下沉直到叶子节点或未完全扩展的节点
        while current.children:
            # 如果当前节点还有未尝试的动作，说明还没扩展完 -> 停止下沉，准备 Expand
            if not current.is_fully_expanded():
                return current
            
            # 选择 PUCT 值最大的子节点
            best_child = None
            best_score = -float('inf')
            
            # 预计算父节点参数
            sqrt_parent_visits = math.sqrt(current.visits + 1e-8)
            
            for child in current.children:
                # Q: 平均价值 (归一化到 -1~1 之间，这取决于 evaluate_leaf 的输出)
                if child.visits > 0:
                    q_val = child.wins / child.visits
                else:
                    # FPU (First Play Urgency): 未访问节点给一个较高的 Q 值
                    # 简单策略：假设它是平局偏好 (0.0) 或微小优势
                    # 配合 Prior，高质量节点自然会被选中
                    q_val = 0.0 

                # U: 探索项 (含 Prior)
                # Formula: c * P * sqrt(N_parent) / (1 + N_child)
                c_puct = self.c_puct_init + math.log((current.visits + self.c_puct_base + 1) / self.c_puct_base)
                u_val = c_puct * child.prior * sqrt_parent_visits / (1 + child.visits)
                
                score = q_val + u_val
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            current = best_child # type: ignore
            
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        扩展节点：
        1. (首次) 生成所有合法动作，计算 Gamma -> Prior，存入 untried_moves。
        2. (每次) 从 untried_moves 中弹出一个 Prior 最高的动作，创建子节点。
        """
        # 1. 首次进入：生成并评估所有动作
        if node.untried_moves is None:
            size = node.board.size
            valid_moves = []
            
            # 收集空位
            for x in range(size):
                for y in range(size):
                    if node.board.tiles[x][y].colour is None:
                        valid_moves.append(Move(x, y))
            
            if not valid_moves:
                node.untried_moves = [] # Terminal
                return node

            # --- [MoHex 核心] 计算 Gamma 并归一化 ---
            
            # 确定 "对手的上一手" (用于 Local Pattern)
            # 如果 node 是 root，对手上一手是 self.last_move
            # 如果 node 是深层节点，对手上一手就是 node.move (即 parent 走到 node 的那步)
            opp_move_ctx = node.move
            if node == self.root_node:
                opp_move_ctx = self.last_move
            
            # 批量计算 Gamma
            gammas = self.pattern_recognizer.get_all_gammas(node.board, valid_moves, opp_move_ctx)
            
            # 归一化 Gamma -> Prior (Rho)
            total_gamma = sum(gammas.values()) + 1e-9
            
            weighted_moves = []
            for mv in valid_moves:
                g = gammas.get((mv.x, mv.y), 1.0)
                prior = g / total_gamma
                weighted_moves.append((mv, prior))
            
            # 按 Prior 从小到大排序 (因为 list.pop() 是取尾部元素，所以最大的要在最后)
            weighted_moves.sort(key=lambda x: x[1])
            
            node.untried_moves = weighted_moves

        if not node.untried_moves:
            return node # 无路可走

        # 2. 弹出一个 Prior 最高的动作进行扩展
        move, prior = node.untried_moves.pop()
        
        # 模拟走棋
        new_board = copy.deepcopy(node.board)
        new_board.set_tile_colour(move.x, move.y, node.player)
        
        # 切换玩家
        next_player = Colour.opposite(node.player)
        
        # 创建子节点
        child = MCTSNode(
            board=new_board,
            move=move,
            parent=node,
            player=next_player,
            prior=prior # 注入计算好的 Prior
        )
        
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        模拟/评估：
        使用 ConnectivityEvaluator (Dijkstra) 直接评估叶子节点的静态价值。
        返回值的视角：相对于 node.player (当前局面的行动方) 的优势。
        范围 [-1.0, 1.0]。
        """
        # 如果是终局，直接返回
        if node.untried_moves is not None and not node.untried_moves and not node.children:
            # 检查是否有胜者
            # 这里的逻辑比较简略，严谨的胜负已由 Evaluator 的 INF 处理
            pass

        # 使用 Dijkstra 评估
        # node.player 是当前需要行动的玩家。
        # Evaluator 返回 正数 表示 node.player 优势大。
        score = self.connectivity_evaluator.evaluate_leaf(node.board, node.player)
        return score

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """
        回传结果。
        Args:
            score: 相对于 node.player 的优势分数。
        """
        current = node
        current_score = score # 当前层视角的收益
        
        while current:
            current.visits += 1
            current.wins += current_score
            
            # 向上回溯一层，视角切换 (零和博弈)
            # Child 的收益 X，对于 Parent 来说是 -X
            current_score = -current_score
            current = current.parent

    def _select_best_move(self, root: MCTSNode) -> Optional[Move]:
        """选择访问次数最多的子节点 (Robust Child)"""
        if not root.children:
            return None
        
        # 找出 visits 最大的子节点
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