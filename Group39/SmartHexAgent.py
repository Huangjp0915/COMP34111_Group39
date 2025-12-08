"""
Group39 Hex AI Agent - 主代理类
"""
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from .mcts_engine import MCTSEngine
from .threat_detector import ThreatDetector
from .connectivity_evaluator import ConnectivityEvaluator
from .pattern_recognizer import PatternRecognizer
from .time_manager import TimeManager
# from .neural_evaluator import NeuralEvaluator
from .utils import logger


class SmartHexAgent(AgentBase):
    """
    Group39 Hex AI代理
    
    核心特性：
    - MCTS（PUCT）搜索
    - 威胁检测优先
    - 连接性评估
    - 智能时间管理
    """
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        
        # 初始化各个模块
        self.mcts_engine = MCTSEngine(colour)
        self.threat_detector = ThreatDetector(colour)
        self.connectivity_evaluator = ConnectivityEvaluator()
        self.pattern_recognizer = PatternRecognizer(colour)
        self.time_manager = TimeManager()
        self.time_manager.start_timer()  # 开始计时
        # self.neural_evaluator = NeuralEvaluator(colour)
        
        # 游戏状态跟踪
        self.last_move = None
        self.turn_count = 0
        self.has_swapped = False
    
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        主决策函数
        
        Args:
            turn: 当前回合数
            board: 当前棋盘状态
            opp_move: 对手的上一手走法（None表示第一手）
        
        Returns:
            Move: 选择的走法
        """
        self.turn_count = turn
        
        try:
            # 0. 时间安全检查
            remaining_time = self.time_manager.get_remaining_time()
            if remaining_time < 0.05:  # 少于50ms
                return self._get_quick_move(board)
            
            # 1. 处理对手的交换（如果对手在turn==2交换了，游戏引擎会自动改变我们的颜色）
            # 但我们需要更新has_swapped状态
            if opp_move and opp_move.is_swap():
                self.has_swapped = True
                # 游戏引擎已经改变了我们的颜色，但我们需要更新所有模块
                self.mcts_engine.colour = self.colour
                self.threat_detector.colour = self.colour
                self.pattern_recognizer.colour = self.colour
                # 重置MCTS根节点和状态（因为颜色改变了，棋盘状态也改变了）
                self.mcts_engine.root_node = None
                self.mcts_engine.last_move = None
                self.mcts_engine.last_board_hash = None
            
            # 2. 交换决策（第二回合，且对手没有交换）
            if turn == 2 and not self.has_swapped and (not opp_move or not opp_move.is_swap()):
                swap_move = self._decide_swap(board, opp_move)
                if swap_move.x == -1 and swap_move.y == -1:
                    # 交换后，更新颜色（游戏引擎会处理，但我们也要更新模块）
                    self.has_swapped = True
                    # 注意：游戏引擎会在_make_move中改变颜色，这里我们只是标记
                return swap_move
            
            # 3. 威胁检测（最高优先级）
            immediate_threats = self.threat_detector.detect_immediate_threats(
                board, self.opp_colour()
            )
            
            # 3.1 如果有必下的点，直接返回
            for x, y, threat_type in immediate_threats:
                if threat_type == "WIN":
                    return Move(x, y)  # 立即获胜（最高优先级）
                elif threat_type == "LOSE":
                    return Move(x, y)  # 必须阻止
            
            # 4. 时间分配
            total_turns_remaining = self._estimate_remaining_turns(board)
            time_budget = self.time_manager.allocate_time(
                board, 
                remaining_time,
                total_turns_remaining,
                self.threat_detector,
                self.opp_colour()
            )
            
            # 5. MCTS搜索
            try:
                # 更新MCTS引擎的last_move（用于根节点复用）
                if opp_move and not opp_move.is_swap():
                    self.mcts_engine.last_move = opp_move
                else:
                    # 如果对手交换了，重置last_move和根节点
                    self.mcts_engine.last_move = None
                    self.mcts_engine.root_node = None
                    self.mcts_engine.last_board_hash = None
                
                move = self.mcts_engine.search(board, time_budget)
                
                # 三重验证走法合法性（确保万无一失）
                if not move:
                    logger.warning("MCTS returned None, using fallback")
                    return self._get_fallback_move(board)
                
                # 验证1：基本合法性
                if not self._is_valid_move(move, board):
                    logger.warning(f"MCTS returned invalid move {move}, using fallback")
                    return self._get_fallback_move(board)
                
                # 验证2：位置是否为空（防御性检查）
                if move.x != -1 and move.y != -1:
                    if board.tiles[move.x][move.y].colour is not None:
                        logger.warning(f"MCTS returned move to occupied position ({move.x}, {move.y}), using fallback")
                        return self._get_fallback_move(board)
                
                # 验证3：坐标范围
                if move.x != -1 and move.y != -1:
                    if not (0 <= move.x < board.size and 0 <= move.y < board.size):
                        logger.warning(f"MCTS returned move out of bounds ({move.x}, {move.y}), using fallback")
                        return self._get_fallback_move(board)
                
                # 所有验证通过
                self.last_move = move
                return move
            except Exception as e:
                logger.error(f"MCTS search failed: {e}")
                import traceback
                traceback.print_exc()
                return self._get_fallback_move(board)
            
        except Exception as e:
            # 任何异常都返回一个合法走法
            logger.error(f"Error in make_move: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_move(board)
    
    def _decide_swap(self, board: Board, opp_move: Move | None) -> Move:
        """
        第二回合的交换决策
        使用连接性评估判断是否交换
        
        Args:
            board: 当前棋盘状态
            opp_move: 对手的第一手走法
        
        Returns:
            Move: 交换(-1, -1)或正常走法
        """
        if self.has_swapped:
            # 已经交换过，不能再交换
            return self._get_first_valid_move(board)
        
        try:
            # 使用连接性评估判断是否交换
            red_cost = self.connectivity_evaluator.shortest_path_cost(board, Colour.RED)
            blue_cost = self.connectivity_evaluator.shortest_path_cost(board, Colour.BLUE)
            
            if self.colour == Colour.RED:
                # RED的视角：如果BLUE成本更低，应该交换
                my_advantage = blue_cost - red_cost
            else:
                # BLUE的视角：如果RED成本更低，应该交换
                my_advantage = red_cost - blue_cost
            
            # 如果对手位置更有利（advantage < 0），交换
            if my_advantage < 0:
                self.has_swapped = True
                logger.info("Agent decided to SWAP")
                return Move(-1, -1)  # 交换
            else:
                # 不交换，正常走法
                return self._get_first_valid_move(board)
        except Exception as e:
            logger.error(f"Error in swap decision: {e}")
            # 出错时默认不交换
            return self._get_first_valid_move(board)
    
    def _get_fallback_move(self, board: Board) -> Move:
        """
        紧急情况下的备用走法
        
        Args:
            board: 当前棋盘状态
        
        Returns:
            Move: 一个合法的走法
        """
        try:
            # 方法1：返回最近访问次数最多的子节点（如果有MCTS根节点）
            # 但必须验证根节点状态与当前棋盘一致
            if self.mcts_engine.root_node:
                # 验证根节点状态
                from .utils import hash_board
                root_hash = hash_board(self.mcts_engine.root_node.board)
                current_hash = hash_board(board)
                if root_hash == current_hash and self.mcts_engine.root_node.children:
                    best_child = max(
                        self.mcts_engine.root_node.children, 
                        key=lambda c: c.visits
                    )
                    if best_child.move and self._is_valid_move(best_child.move, board):
                        # 再次验证位置是否为空
                        if best_child.move.x != -1 and best_child.move.y != -1:
                            if board.tiles[best_child.move.x][best_child.move.y].colour is None:
                                return best_child.move
            
            # 方法2：返回模式识别推荐的点
            patterns = self.pattern_recognizer.detect_simple_patterns(board, self.colour)
            if patterns:
                # 按权重排序，选择权重最高的
                patterns.sort(key=lambda p: p[2], reverse=True)
                for x, y, _ in patterns:
                    move = Move(x, y)
                    if self._is_valid_move(move, board):
                        # 再次验证位置是否为空
                        if board.tiles[x][y].colour is None:
                            return move
            
            # 方法3：返回第一个合法走法（最安全）
            return self._get_first_valid_move(board)
        except Exception as e:
            logger.error(f"Error in fallback move: {e}")
            # 最后的备用方案
            return self._get_first_valid_move(board)
    
    def _get_quick_move(self, board: Board) -> Move:
        """
        时间过少时的快速走法
        只检查威胁，不进行MCTS搜索
        
        Args:
            board: 当前棋盘状态
        
        Returns:
            Move: 一个合法的走法
        """
        try:
            # 只检查威胁，不进行MCTS搜索
            immediate_threats = self.threat_detector.detect_immediate_threats(
                board, self.opp_colour()
            )
            if immediate_threats:
                for x, y, threat_type in immediate_threats:
                    move = Move(x, y)
                    if self._is_valid_move(move, board):
                        return move
            
            # 如果没有威胁，返回第一个合法走法
            return self._get_first_valid_move(board)
        except Exception as e:
            logger.error(f"Error in quick move: {e}")
            return self._get_first_valid_move(board)
    
    def _get_first_valid_move(self, board: Board) -> Move:
        """
        返回第一个合法走法
        
        Args:
            board: 当前棋盘状态
        
        Returns:
            Move: 第一个空位的走法
        """
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    return Move(x, y)
        
        # 如果棋盘已满（理论上不会发生），返回交换
        return Move(-1, -1)
    
    def _is_valid_move(self, move: Move, board: Board) -> bool:
        """
        检查走法是否合法
        
        Args:
            move: 待检查的走法
            board: 当前棋盘状态
        
        Returns:
            bool: 是否合法
        """
        if move.x == -1 and move.y == -1:
            return True  # 交换走法
        
        if 0 <= move.x < board.size and 0 <= move.y < board.size:
            return board.tiles[move.x][move.y].colour is None
        
        return False
    
    def _estimate_remaining_turns(self, board: Board) -> int:
        """
        估算剩余回合数
        
        Args:
            board: 当前棋盘状态
        
        Returns:
            int: 估算的剩余回合数
        """
        empty_tiles = sum(
            1 for x in range(board.size) 
            for y in range(board.size) 
            if board.tiles[x][y].colour is None
        )
        return max(1, empty_tiles // 2)  # 粗略估算

