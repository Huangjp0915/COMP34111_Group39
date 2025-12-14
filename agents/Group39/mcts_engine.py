"""
MCTS引擎 - PUCT版本
"""
import copy
import math
import time
from typing import Optional

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from .connectivity_evaluator import ConnectivityEvaluator
from .pattern_recognizer import PatternRecognizer
from .phase import GamePhase
from .threat_detector import ThreatDetector
from .utils import count_empty_tiles, hash_board
import random

from .zobrist_hash import ZobristHash


class MCTSNode:
    """
    MCTS节点
    
    属性：
    - board: 棋盘状态
    - move: 到达此节点的走法
    - parent: 父节点
    - children: 子节点列表
    - visits: 访问次数
    - wins: 累计价值（-1到1）
    - untried_moves: 未尝试的走法列表
    - player: 当前玩家
    - prior: 先验概率
    """

    def __init__(self, board: Board, move: Optional[Move] = None,
                 parent: Optional['MCTSNode'] = None, player: Optional[Colour] = None):
        self.board = copy.deepcopy(board)  # 深拷贝棋盘
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0  # 累计价值（-1到1）
        self.player = player
        self.prior = 0.1  # 先验概率（默认值）
        # 新增RAVE用常量
        self.rave_wins = {}  # {(x,y): wins}
        self.rave_visits = {}  # {(x,y): visits}

        # 初始化未尝试的走法列表
        # 优化：初始化为列表（后续会进行排序）
        self.untried_moves = []
        if self.player:
            for x in range(board.size):
                for y in range(board.size):
                    if board.tiles[x][y].colour is None:
                        self.untried_moves.append(Move(x, y))

    def is_fully_expanded(self) -> bool:
        """检查节点是否完全展开"""
        return len(self.untried_moves) == 0 and len(self.children) > 0

    def is_terminal(self) -> bool:
        """检查节点是否为终局"""
        if not self.player:
            return False
        return (self.board.has_ended(self.player) or
                self.board.has_ended(Colour.opposite(self.player)))


class MCTSEngine:
    """
    MCTS搜索引擎（PUCT版本）
    
    特性：
    - PUCT公式（而非UCB1）
    - 动态PUCT系数
    - 根节点复用
    - 时间循环（而非固定迭代）
    """

    def __init__(self, colour: Colour, board_size: int = 11):
        self.colour = colour

        # -------------------------------------------------
        # Global search state (TT-based architecture)
        # -------------------------------------------------
        self.root_node: Optional[MCTSNode] = None

        # Zobrist hashing (global, shared)
        self.zobrist = ZobristHash(size=board_size)

        # Transposition Table: hash -> MCTSNode
        self.tt = {}

        # -------------------------------------------------
        # Heuristic / evaluation modules
        # -------------------------------------------------
        self.connectivity_evaluator = ConnectivityEvaluator()
        self.pattern_recognizer = PatternRecognizer(colour)
        self.threat_detector = ThreatDetector(colour)

    def _apply_root_noise(self, root: MCTSNode,
                          epsilon: float = 0.25,
                          noise_scale: float = 0.3):
        """
        Apply Dirichlet-like noise to root priors.
        Only affects exploration at the root.
        """
        if not root.children:
            return

        # Simple Dirichlet approximation: random noise per child
        noises = [random.random() for _ in root.children]
        total = sum(noises)
        noises = [n / total for n in noises]

        for child, noise in zip(root.children, noises):
            child.prior = (
                    (1 - epsilon) * child.prior +
                    epsilon * noise * noise_scale
            )

    def search(self, board: Board, time_budget: float) -> Move:
        current_hash = self.zobrist.get_hash(board, self.colour)

        # --- Root lookup via Transposition Table ---
        if current_hash in self.tt:
            print("[TT HIT]")
            self.root_node = self.tt[current_hash]
            self.root_node.parent = None
            # Debug safety check (optional but recommended)
            # assert self.root_node.player == self.colour
        else:
            self.root_node = MCTSNode(board, player=self.colour)
            self.root_node.zobrist_hash = current_hash
            self.tt[current_hash] = self.root_node

        self._apply_root_noise(self.root_node)

        # --- MCTS main loop ---
        start_time = time.perf_counter()
        iteration = 0

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_budget * 0.95:
                break

            self._mcts_iteration(self.root_node)
            iteration += 1

            if iteration > 100000:
                break

        # --- Final move selection ---
        if self.root_node and self.root_node.children:
            valid_children = [
                child for child in self.root_node.children
                if child.move and self._is_valid_move_for_board(child.move, board)
            ]

            if valid_children:
                best_child = max(valid_children, key=lambda c: c.visits)
                best_move = best_child.move
                if best_move and self._is_valid_move_for_board(best_move, board):
                    return best_move

        return self._get_first_valid_move(board)

    def _is_valid_move_for_board(self, move: Move, board: Board) -> bool:
        """检查走法对当前棋盘是否合法"""
        if move.x == -1 and move.y == -1:
            return True  # 交换走法
        if 0 <= move.x < board.size and 0 <= move.y < board.size:
            return board.tiles[move.x][move.y].colour is None
        return False

    def _mcts_iteration(self, root: MCTSNode):
        """
        MCTS单次迭代：Selection -> Expansion -> Simulation -> Backpropagation
        
        Args:
            root: 根节点
        """
        # 1. Selection: 从根节点选择到叶子节点
        node = self._select(root)

        # 2. Expansion: 如果未完全展开，添加新子节点
        if node.untried_moves:
            node = self._expand(node)

        # 3. Simulation: 从新节点模拟到终局
        result_amaf = self._simulate(node)

        # 4. Backpropagation: 将结果回传到根节点
        self._backpropagate(node, result_amaf)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection using PUCT + RAVE (AMAF).

        Traverses the tree until:
        - a node with untried moves, or
        - a leaf node
        """
        while node.children and not node.untried_moves:

            best_child = None
            best_score = float('-inf')

            parent_visits = max(1, node.visits)

            for child in node.children:

                # ---- 1. Normal Q value ----
                if child.visits > 0:
                    Q = child.wins / child.visits
                else:
                    Q = 0.0

                # ---- 2. RAVE / AMAF Q value ----
                move = child.move
                if move:
                    move_xy = (move.x, move.y)
                else:
                    move_xy = None

                if move_xy and move_xy in node.rave_visits:
                    rave_v = node.rave_visits[move_xy]
                    if rave_v > 0:
                        Q_amaf = node.rave_wins[move_xy] / rave_v
                    else:
                        Q_amaf = 0.0
                else:
                    rave_v = 0
                    Q_amaf = 0.0

                # ---- 3. Beta: trust RAVE early, trust real stats later ----
                beta = rave_v / (child.visits + rave_v + 1e-6)

                Q_mix = (1.0 - beta) * Q + beta * Q_amaf

                # ---- 4. Exploration term (PUCT) ----
                c_puct = self._calculate_dynamic_c_puct(node.board, child)

                U = (
                        c_puct
                        * child.prior
                        * (parent_visits ** 0.5)
                        / (1 + child.visits)
                )

                score = Q_mix + U

                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                break

            node = best_child

        return node

    def _calculate_puct(self, node: MCTSNode, c_puct: float) -> float:
        if node.visits == 0:
            return float('inf')

        # --- base Q ---
        Q = node.wins / node.visits

        # --- RAVE Q ---
        move_xy = (node.move.x, node.move.y)
        parent = node.parent

        if parent and move_xy in parent.rave_visits and parent.rave_visits[move_xy] > 0:
            Q_rave = parent.rave_wins[move_xy] / parent.rave_visits[move_xy]
        else:
            Q_rave = Q  # fallback

        # --- RAVE blending ---
        # k 是调节参数，Hex 通常设置为 300 ~ 1000
        k = 300
        beta = k / (node.visits + k)

        Q_hat = (1 - beta) * Q + beta * Q_rave

        # --- PUCT formula using Q_hat ---
        P = node.prior
        N = parent.visits if parent else 1
        n = node.visits

        return Q_hat + c_puct * P * math.sqrt(N) / (1 + n)

    def _calculate_dynamic_c_puct(self, board: Board, node: MCTSNode) -> float:
        """
        计算动态PUCT系数
        
        Args:
            board: 棋盘状态
            node: 节点
        
        Returns:
            float: 动态探索系数
        """
        base_c = 1.5  # 基础探索系数

        # 根据游戏阶段调整
        empty_tiles = count_empty_tiles(board)
        total_tiles = board.size * board.size
        game_phase = empty_tiles / total_tiles if total_tiles > 0 else 0.5

        if game_phase > 0.7:  # 早期：更多探索
            phase_factor = 1.2
        elif game_phase < 0.3:  # 后期：更多利用
            phase_factor = 0.8
        else:  # 中期
            phase_factor = 1.0

        # 根据Q值调整
        if node.visits > 0:
            q_value = node.wins / node.visits
            if abs(q_value) > 0.7:  # 价值很明确，减少探索
                q_factor = 0.8
            else:  # 价值不明确，保持探索
                q_factor = 1.0
        else:
            q_factor = 1.0

        return base_c * phase_factor * q_factor

    def _is_deadish_move(self, board: Board, move: Move, player: Colour) -> bool:
        """
        Cheap dead-ish heuristic:
        A move is dead-ish if:
        - it does not improve my connectivity
        - AND does not worsen opponent connectivity
        """

        opp = Colour.opposite(player)

        # --- before ---
        my_before = self.connectivity_evaluator.shortest_path_cost(board, player)
        opp_before = self.connectivity_evaluator.shortest_path_cost(board, opp)

        # --- simulate move ---
        board.tiles[move.x][move.y].colour = player

        my_after = self.connectivity_evaluator.shortest_path_cost(board, player)
        opp_after = self.connectivity_evaluator.shortest_path_cost(board, opp)

        # --- undo ---
        board.tiles[move.x][move.y].colour = None

        # --- dead-ish condition ---
        # no benefit for me AND no harm to opponent
        if my_after >= my_before and opp_after <= opp_before:
            return True

        return False

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        TT-aware expansion:
        - Reuses states via Zobrist + TT
        - Allows multiple parents (DAG)
        - Filters dead-ish moves
        """

        if not node.untried_moves:
            return node

        # ------------------------------------------------------------
        # 1. Filter legal untried moves
        # ------------------------------------------------------------
        valid_moves = []
        for candidate in node.untried_moves:
            if (0 <= candidate.x < node.board.size and
                    0 <= candidate.y < node.board.size and
                    node.board.tiles[candidate.x][candidate.y].colour is None):
                valid_moves.append(candidate)

        if not valid_moves:
            return node

        # ------------------------------------------------------------
        # 2. Dead-ish move filtering (cheap heuristic)
        #    Only apply when branching factor is large
        # ------------------------------------------------------------
        if len(valid_moves) > 20:
            filtered = []
            for move in valid_moves:
                if not self._is_deadish_move(node.board, move, node.player):
                    filtered.append(move)
            if filtered:
                valid_moves = filtered

        # ------------------------------------------------------------
        # 3. Score moves (prior + heuristic)
        # ------------------------------------------------------------
        move_scores = []
        phase = self.get_game_phase(node.board)
        for move in valid_moves:
            prior = self.pattern_recognizer.get_prior(
                node.board, move, node.player, phase
            )
            heuristic_score = self._calculate_move_heuristic(
                node.board, move, node.player
            )
            total_score = 0.6 * prior + 0.4 * heuristic_score
            move_scores.append((move, total_score))

        move_scores.sort(key=lambda x: x[1], reverse=True)

        # ------------------------------------------------------------
        # 4. Weighted random selection
        # ------------------------------------------------------------
        if len(move_scores) > 3:
            top_moves = move_scores[:max(1, len(move_scores) // 3)]
            if random.random() < 0.7:
                move = random.choice(top_moves)[0]
            else:
                move = random.choice(move_scores)[0]
        else:
            move = move_scores[0][0]

        # ------------------------------------------------------------
        # 5. Remove chosen move from untried
        # ------------------------------------------------------------
        if move in node.untried_moves:
            node.untried_moves.remove(move)

        # ------------------------------------------------------------
        # 6. Create new board state
        # ------------------------------------------------------------
        new_board = copy.deepcopy(node.board)
        new_board.set_tile_colour(move.x, move.y, node.player)

        next_player = Colour.opposite(node.player)

        # ------------------------------------------------------------
        # 7. TT lookup (state = board + side-to-move)
        # ------------------------------------------------------------
        new_hash = self.zobrist.get_hash(new_board, next_player)

        if new_hash in self.tt:
            child = self.tt[new_hash]
        else:
            child = MCTSNode(
                board=new_board,
                move=move,
                parent=None,  # path-parent set below
                player=next_player
            )
            child.zobrist_hash = new_hash

            # Initialize prior ONCE for new state
            child.prior = self.pattern_recognizer.get_prior(
                node.board, move, node.player, phase
            )

            self.tt[new_hash] = child

        # ------------------------------------------------------------
        # 8. Attach child to current node (graph edge)
        # ------------------------------------------------------------
        if child not in node.children:
            node.children.append(child)

        # ------------------------------------------------------------
        # 9. Set path-parent for backpropagation
        # ------------------------------------------------------------
        child.parent = node

        return child

    def _simulate(self, node: MCTSNode):
        """
        Phase-aware heavy playout with heuristics + AMAF recording.

        Returns:
            (result, amaf_moves)
            result: [-1, 1] from ROOT player's perspective
            amaf_moves: [(player, (x,y)), ...]
        """

        # --- Terminal check ---
        if node.is_terminal():
            if node.board.has_ended(self.colour):
                return 1.0, []
            elif node.board.has_ended(Colour.opposite(self.colour)):
                return -1.0, []
            else:
                return 0.0, []

        board = copy.deepcopy(node.board)
        player = node.player
        amaf_moves = []

        # --------------------------------------------------
        # Unified phase (CRITICAL)
        # --------------------------------------------------
        phase = self.get_game_phase(board)

        # --------------------------------------------------
        # OPENING: no rollout, no AMAF
        # --------------------------------------------------
        if phase == GamePhase.OPENING:
            value = self.connectivity_evaluator.evaluate_leaf(board, self.colour)
            return value, []

        # --------------------------------------------------
        # Decide rollout depth + frequency by phase
        # --------------------------------------------------
        if phase == GamePhase.MIDGAME:
            MAX_PLAYOUT_STEPS = 2
            heavy_prob = 0.3
        else:  # LATEGAME
            MAX_PLAYOUT_STEPS = 4
            heavy_prob = 0.5

        # Frequency gate
        if random.random() > heavy_prob:
            value = self.connectivity_evaluator.evaluate_leaf(board, self.colour)
            return value, []

        # --------------------------------------------------
        # Heavy playout (controlled)
        # --------------------------------------------------
        for step in range(MAX_PLAYOUT_STEPS):

            # --- Early terminal check ---
            if board.has_ended(self.colour):
                return 1.0, amaf_moves
            if board.has_ended(Colour.opposite(self.colour)):
                return -1.0, amaf_moves

            # --- Collect legal moves ---
            legal_moves = [
                (x, y)
                for x in range(board.size)
                for y in range(board.size)
                if board.tiles[x][y].colour is None
            ]

            if not legal_moves:
                break

            chosen_move = None

            # --- 1. Threat check (1-ply win / lose) ---
            threats = self.threat_detector.detect_immediate_threats(board, player)
            if threats:
                win_moves = [m for m in threats if m[2] == "WIN"]
                lose_moves = [m for m in threats if m[2] == "LOSE"]

                if win_moves:
                    chosen_move = (win_moves[0][0], win_moves[0][1])
                elif lose_moves:
                    chosen_move = (lose_moves[0][0], lose_moves[0][1])

            # --- 2. Pattern / heuristic-guided move ---
            if chosen_move is None:
                scored_moves = []
                for (x, y) in legal_moves:
                    move = Move(x, y)
                    prior = self.pattern_recognizer.get_prior(board, move, player, phase)
                    scored_moves.append(((x, y), prior))

                scored_moves.sort(key=lambda x: x[1], reverse=True)

                TOP_K = max(1, len(scored_moves) // 4)
                if random.random() < 0.7:
                    chosen_move = random.choice(scored_moves[:TOP_K])[0]
                else:
                    chosen_move = random.choice(scored_moves)[0]

            mx, my = chosen_move

            # --- Record AMAF (VERY shallow) ---
            if step < 2:
                amaf_moves.append((player, (mx, my)))

            # --- Play move ---
            board.set_tile_colour(mx, my, player)
            player = Colour.opposite(player)

            # --- Optional early cutoff ---
            eval_score = self.connectivity_evaluator.evaluate_leaf(board, self.colour)
            if abs(eval_score) > 0.8:
                return eval_score, amaf_moves

        # --- Final static evaluation ---
        result = self.connectivity_evaluator.evaluate_leaf(board, self.colour)
        return result, amaf_moves

    def get_game_phase(self, board: Board) -> GamePhase:
        empty = sum(
            1 for x in range(board.size)
            for y in range(board.size)
            if board.tiles[x][y].colour is None
        )
        empty_ratio = empty / (board.size * board.size)

        if empty_ratio > 0.75:
            return GamePhase.OPENING
        elif empty_ratio > 0.35:
            return GamePhase.MIDGAME
        else:
            return GamePhase.LATEGAME

    def _heuristic_simulation(self, node: MCTSNode, max_steps: int = 3) -> float:
        """
        启发式模拟：使用启发式走法选择进行有限步数模拟
        
        Args:
            node: 当前节点
            max_steps: 最大模拟步数
        
        Returns:
            float: 模拟结果（-1到1）
        """
        current_board = copy.deepcopy(node.board)
        current_player = node.player
        steps = 0

        # 进行有限步数的启发式模拟
        while steps < max_steps:
            # 检查是否终局
            if current_board.has_ended(self.colour):
                return 1.0
            elif current_board.has_ended(Colour.opposite(self.colour)):
                return -1.0

            # 获取所有合法走法
            legal_moves = []
            for x in range(current_board.size):
                for y in range(current_board.size):
                    if current_board.tiles[x][y].colour is None:
                        legal_moves.append(Move(x, y))

            if not legal_moves:
                break

            # 使用启发式选择走法（而不是完全随机）
            best_move = self._select_heuristic_move(current_board, legal_moves, current_player)

            # 执行走法
            current_board.set_tile_colour(best_move.x, best_move.y, current_player)
            current_player = Colour.opposite(current_player)
            steps += 1

        # 使用连接性评估作为最终评估
        return self.connectivity_evaluator.evaluate_leaf(current_board, self.colour)

    def _select_heuristic_move(self, board: Board, legal_moves: list[Move], player: Colour) -> Move:
        """
        启发式选择走法：基于连接性和模式识别
        
        Args:
            board: 当前棋盘
            legal_moves: 合法走法列表
            player: 当前玩家
        
        Returns:
            Move: 选择的走法
        """
        if not legal_moves:
            return None

        # 计算每个走法的启发式评分
        move_scores = []
        phase = self.get_game_phase(board)
        for move in legal_moves:
            # 1. 模式识别评分
            pattern_score = self.pattern_recognizer.get_prior(board, move, player, phase)

            # 2. 连接性启发式
            connectivity_score = self._calculate_move_heuristic(board, move, player)

            # 3. 综合评分
            total_score = 0.5 * pattern_score + 0.5 * connectivity_score
            move_scores.append((move, total_score))

        # 按评分排序
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # 使用加权随机选择（前50%有更高概率）
        if len(move_scores) > 2:
            top_count = max(1, len(move_scores) // 2)
            top_moves = move_scores[:top_count]
            if random.random() < 0.8:  # 80%概率选择前50%
                return random.choice(top_moves)[0]
            else:
                return random.choice(move_scores)[0]
        else:
            return move_scores[0][0]

    def _calculate_move_heuristic(self, board: Board, move: Move, player: Colour) -> float:
        """
        计算走法的连接性启发式评分
        
        Args:
            board: 当前棋盘
            move: 走法
            player: 玩家
        
        Returns:
            float: 启发式评分（0到1）
        """
        # 模拟下子
        test_board = copy.deepcopy(board)
        test_board.set_tile_colour(move.x, move.y, player)

        # 计算下子前后的连接性改善
        before_cost = self.connectivity_evaluator.shortest_path_cost(board, player)
        after_cost = self.connectivity_evaluator.shortest_path_cost(test_board, player)

        # 计算对手的连接性变化
        opp_before_cost = self.connectivity_evaluator.shortest_path_cost(board, Colour.opposite(player))
        opp_after_cost = self.connectivity_evaluator.shortest_path_cost(test_board, Colour.opposite(player))

        # 评分：自己的连接性改善 - 对手的连接性改善
        my_improvement = before_cost - after_cost if before_cost != float('inf') else 0
        opp_improvement = opp_before_cost - opp_after_cost if opp_before_cost != float('inf') else 0

        # 归一化到0-1范围
        # 假设最大改善为board.size（连接一条完整路径）
        max_improvement = board.size
        normalized_score = (my_improvement - opp_improvement) / (2 * max_improvement) + 0.5
        return max(0.0, min(1.0, normalized_score))

    def _backpropagate(self, node: MCTSNode, result_amaf):
        result, amaf_moves = result_amaf

        current = node

        while current:

            # --- normal MCTS update ---
            current.visits += 1

            if current.player == self.colour:
                value = result
            else:
                value = -result
            current.wins += value

            # --- RAVE update ---
            for player, move_xy in amaf_moves:
                # 只更新“与当前节点同色玩家”的动作（Hex 必须这么做）
                if player != current.player:
                    continue

                if move_xy not in current.rave_visits:
                    current.rave_visits[move_xy] = 0
                    current.rave_wins[move_xy] = 0.0

                current.rave_visits[move_xy] += 1
                if value > 0:
                    current.rave_wins[move_xy] += 1

            current = current.parent

    def _select_best_move(self, node: MCTSNode) -> Move:
        """
        选择最优走法（访问次数最多）
        
        Args:
            node: 根节点
        
        Returns:
            Move: 最优走法
        """
        if not node.children:
            return self._get_first_valid_move(node.board)

        # 选择访问次数最多的子节点
        best_child = max(node.children, key=lambda c: c.visits)
        if best_child.move:
            return best_child.move
        else:
            return self._get_first_valid_move(node.board)

    def _get_first_valid_move(self, board: Board) -> Move:
        """返回第一个合法走法"""
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    return Move(x, y)
        return Move(-1, -1)

    def _find_matching_child(self, root: MCTSNode, move: Move) -> Optional[MCTSNode]:
        """
        在根节点的子节点中查找匹配的走法（用于根节点复用）
        优化：更精确的匹配，考虑棋盘状态一致性
        
        Args:
            root: 根节点
            move: 对手的走法
        
        Returns:
            Optional[MCTSNode]: 匹配的子节点，如果找不到返回None
        """
        if not root or not root.children:
            return None

        # 优化：优先匹配访问次数多、价值高的子节点
        # 这样可以复用更有价值的子树
        candidates = []
        for child in root.children:
            if child.move and child.move.x == move.x and child.move.y == move.y:
                # 计算匹配质量评分（访问次数 + 价值）
                quality_score = child.visits
                if child.visits > 0:
                    quality_score += abs(child.wins / child.visits) * 10
                candidates.append((child, quality_score))

        if not candidates:
            return None

        # 选择质量最高的匹配
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _boards_match(self, board1: Board, board2: Board) -> bool:
        """
        验证两个棋盘状态是否完全一致
        
        Args:
            board1: 第一个棋盘
            board2: 第二个棋盘
        
        Returns:
            bool: 是否一致
        """
        if board1.size != board2.size:
            return False

        for x in range(board1.size):
            for y in range(board1.size):
                if board1.tiles[x][y].colour != board2.tiles[x][y].colour:
                    return False

        return True
