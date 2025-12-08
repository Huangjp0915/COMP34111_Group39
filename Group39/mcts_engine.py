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
from .utils import count_empty_tiles, hash_board
import random


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
    
    def __init__(self, colour: Colour):
        self.colour = colour
        self.root_node: Optional[MCTSNode] = None
        self.last_board_hash = None
        self.last_move: Optional[Move] = None
        
        # 初始化其他模块
        self.connectivity_evaluator = ConnectivityEvaluator()
        self.pattern_recognizer = PatternRecognizer(colour)
    
    def search(self, board: Board, time_budget: float) -> Move:
        """
        执行MCTS搜索
        
        Args:
            board: 当前棋盘状态
            time_budget: 时间预算（秒）
        
        Returns:
            Move: 最优走法
        """
        # 强制验证：根节点必须与当前棋盘状态完全一致
        # 如果不一致，直接创建新根节点
        current_board_hash = hash_board(board)
        
        # 验证根节点是否与当前棋盘状态一致
        root_valid = False
        if self.root_node:
            # 检查根节点的棋盘状态是否与当前棋盘一致
            root_hash = hash_board(self.root_node.board)
            if root_hash == current_board_hash:
                # 进一步验证：检查棋盘上的每个位置是否一致
                if self._boards_match(self.root_node.board, board):
                    root_valid = True
        
        if root_valid and self.last_move:
            # 尝试找到匹配的子节点
            matching_child = self._find_matching_child(
                self.root_node, self.last_move
            )
            if matching_child:
                # 验证匹配子节点的棋盘状态
                child_hash = hash_board(matching_child.board)
                if child_hash == current_board_hash and self._boards_match(matching_child.board, board):
                    self.root_node = matching_child
                    self.root_node.parent = None  # 成为新根节点
                else:
                    # 棋盘状态不一致，创建新根节点
                    self.root_node = MCTSNode(board, player=self.colour)
            else:
                # 如果找不到匹配的子节点，创建新根节点
                self.root_node = MCTSNode(board, player=self.colour)
        else:
            # 第一次搜索或没有历史，创建新根节点
            self.root_node = MCTSNode(board, player=self.colour)
        
        self.last_board_hash = current_board_hash
        
        # 按时间循环搜索
        start_time = time.perf_counter()
        iteration = 0
        
        try:
            while True:
                # 时间检查
                elapsed = time.perf_counter() - start_time
                if elapsed >= time_budget * 0.95:  # 预留5%缓冲
                    break
                
                # MCTS迭代
                self._mcts_iteration(self.root_node)
                iteration += 1
                
                # 安全网：防止无限循环
                if iteration > 100000:
                    break
        except Exception as e:
            pass
        
        # 选择最优走法
        # 重要：在返回走法前，再次验证根节点状态与当前棋盘一致
        # 如果根节点状态不一致，直接返回第一个合法走法
        if self.root_node:
            root_hash = hash_board(self.root_node.board)
            current_hash = hash_board(board)
            if root_hash != current_hash or not self._boards_match(self.root_node.board, board):
                # 根节点状态不一致，返回第一个合法走法
                return self._get_first_valid_move(board)
        
        if self.root_node and self.root_node.children:
            # 从所有子节点中选择，但只考虑合法的走法
            valid_children = [
                child for child in self.root_node.children
                if child.move and self._is_valid_move_for_board(child.move, board)
            ]
            
            if valid_children:
                # 选择访问次数最多的合法子节点
                best_child = max(valid_children, key=lambda c: c.visits)
                best_move = best_child.move
                # 再次验证走法合法性
                if best_move and self._is_valid_move_for_board(best_move, board):
                    self.last_move = best_move
                    return best_move
                else:
                    # 走法不合法，返回第一个合法走法
                    return self._get_first_valid_move(board)
            else:
                # 所有子节点的走法都不合法，返回第一个合法走法
                return self._get_first_valid_move(board)
        
        # 如果没有子节点，返回第一个合法走法
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
        result = self._simulate(node)
        
        # 4. Backpropagation: 将结果回传到根节点
        self._backpropagate(node, result)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        选择：使用PUCT公式选择最优子节点
        
        Args:
            node: 当前节点
        
        Returns:
            MCTSNode: 选择的子节点
        """
        while node.children and not node.untried_moves:
            # 计算所有子节点的PUCT值
            best_child = None
            best_puct = float('-inf')
            
            for child in node.children:
                c_puct = self._calculate_dynamic_c_puct(node.board, child)
                puct_value = self._calculate_puct(child, c_puct)
                
                if puct_value > best_puct:
                    best_puct = puct_value
                    best_child = child
            
            if best_child:
                node = best_child
            else:
                break
        
        return node
    
    def _calculate_puct(self, node: MCTSNode, c_puct: float) -> float:
        """
        计算PUCT值
        
        Args:
            node: 子节点
            c_puct: 动态探索系数
        
        Returns:
            float: PUCT值
        """
        if node.visits == 0:
            return float('inf')  # 未访问的节点优先
        
        Q = node.wins / node.visits  # 平均价值
        P = node.prior  # 先验概率
        N = node.parent.visits if node.parent else 1  # 父节点访问次数
        n = node.visits  # 当前节点访问次数
        
        puct = Q + c_puct * P * math.sqrt(N) / (1 + n)
        return puct
    
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
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        扩展：添加一个新子节点
        优化：基于prior和连接性对移动进行排序，优先选择更有希望的走法
        
        Args:
            node: 当前节点
        
        Returns:
            MCTSNode: 新创建的子节点
        """
        if not node.untried_moves:
            return node
        
        # 优化：对未尝试的走法进行排序（基于prior和连接性）
        # 过滤出合法的走法
        valid_moves = []
        for candidate in node.untried_moves:
            if (0 <= candidate.x < node.board.size and 
                0 <= candidate.y < node.board.size and
                node.board.tiles[candidate.x][candidate.y].colour is None):
                valid_moves.append(candidate)
        
        if not valid_moves:
            # 没有合法的未尝试走法
            return node
        
        # 计算每个走法的评分（prior + 连接性启发式）
        move_scores = []
        for move in valid_moves:
            # 1. 获取prior（来自模式识别）
            prior = self.pattern_recognizer.get_prior(node.board, move, node.player)
            
            # 2. 计算连接性启发式（模拟下子后的连接性改善）
            heuristic_score = self._calculate_move_heuristic(node.board, move, node.player)
            
            # 3. 综合评分（prior权重0.6，启发式权重0.4）
            total_score = 0.6 * prior + 0.4 * heuristic_score
            move_scores.append((move, total_score))
        
        # 按评分排序（降序）
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 使用加权随机选择（而不是完全随机）
        # 前30%的走法有更高的概率被选中
        if len(move_scores) > 3:
            # 前30%的走法
            top_moves = move_scores[:max(1, len(move_scores) // 3)]
            # 70%概率选择前30%，30%概率随机选择
            if random.random() < 0.7:
                move = random.choice(top_moves)[0]
            else:
                move = random.choice(move_scores)[0]
        else:
            # 走法较少时，直接选择评分最高的
            move = move_scores[0][0]
        
        # 从untried_moves中移除选中的走法
        if move in node.untried_moves:
            node.untried_moves.remove(move)
        
        # 创建新棋盘状态
        new_board = copy.deepcopy(node.board)
        new_board.set_tile_colour(move.x, move.y, node.player)
        
        # 创建新节点
        new_player = Colour.opposite(node.player)
        child = MCTSNode(
            board=new_board,
            move=move,
            parent=node,
            player=new_player
        )
        
        # 初始化prior（来自模式识别）
        child.prior = self.pattern_recognizer.get_prior(
            node.board, move, node.player
        )
        
        node.children.append(child)
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        模拟：从当前节点模拟到终局（使用启发式模拟策略）
        优化：使用启发式走法选择，而不是完全随机
        
        Args:
            node: 当前节点
        
        Returns:
            float: 模拟结果（-1到1）
        """
        # 如果已经是终局，直接返回结果
        if node.is_terminal():
            if node.board.has_ended(self.colour):
                return 1.0
            elif node.board.has_ended(Colour.opposite(self.colour)):
                return -1.0
            else:
                return 0.0
        
        # 优化：使用启发式模拟（而不是完全随机）
        # 根据游戏阶段决定是否使用启发式模拟
        # 为了平衡质量和速度，大部分情况下直接使用连接性评估
        # 只在特定情况下使用启发式模拟（例如：早期且时间充足）
        empty_tiles = count_empty_tiles(node.board)
        total_tiles = node.board.size * node.board.size
        game_phase = empty_tiles / total_tiles if total_tiles > 0 else 0.5
        
        # 为了性能，大部分情况下直接使用连接性评估
        # 启发式模拟的开销较大，只在必要时使用
        # 直接使用连接性评估
        return self.connectivity_evaluator.evaluate_leaf(node.board, self.colour)
    
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
        for move in legal_moves:
            # 1. 模式识别评分
            pattern_score = self.pattern_recognizer.get_prior(board, move, player)
            
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
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """
        回传：将结果沿路径回传到根节点
        
        Args:
            node: 当前节点（叶子节点）
            result: 模拟结果（从根节点玩家self.colour的视角：1.0=赢，-1.0=输）
        """
        current_node = node
        # result是从根节点玩家（self.colour）的视角
        # 在向上传播时，需要根据每个节点的玩家调整value
        while current_node:
            current_node.visits += 1
            
            # 计算从当前节点玩家视角的value
            # 如果当前节点是根节点玩家，value = result
            # 如果当前节点是对手，value = -result（对手的胜利是我们的失败）
            if current_node.player == self.colour:
                value = result
            else:
                value = -result
            
            current_node.wins += value
            current_node = current_node.parent
    
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

