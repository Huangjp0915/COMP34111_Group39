"""
Group392 Hex AI Agent - 主代理类
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
    Group392 Hex AI代理
    
    核心特性：
    - MCTS（PUCT）搜索
    - 威胁检测优先
    - 连接性评估
    - 智能时间管理
    """

    def __init__(self, colour: Colour, board_size: int = 11):
        super().__init__(colour)

        # -------------------------------------------------
        # Core search engine (TT + Zobrist based)
        # -------------------------------------------------
        self.mcts_engine = MCTSEngine(colour, board_size)

        # -------------------------------------------------
        # Tactical / heuristic modules
        # -------------------------------------------------
        self.threat_detector = ThreatDetector(colour)
        self.pattern_recognizer = PatternRecognizer(colour)
        self.connectivity_evaluator = ConnectivityEvaluator()

        # -------------------------------------------------
        # Time management
        # -------------------------------------------------
        self.time_manager = TimeManager()
        self.time_manager.start_timer()

        # -------------------------------------------------
        # Game state tracking (agent-level only)
        # -------------------------------------------------
        self.last_move = None  # for logging / UI only
        self.turn_count = 0
        self.has_swapped = False

        self._last_colour = self.colour

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        主决策函数
        
        Args:
            turn: 当前回合数
            board: 当前棋盘状态
            opp_move: 对手的上一手走法（None表示第一手）
        
        Returns:
            Move: 选择的走法

        只做策略层与安全层的事，不再干涉搜索内部状态
        """
        self.turn_count = turn
        if turn == 1 and opp_move is None:
            self.time_manager.start_timer()
            self.has_swapped = False
            self.last_move = None
            self.mcts_engine.root_node = None
            self.mcts_engine.tt.clear()

        self.time_manager.begin_move()

        if self.colour != self._last_colour:
            self._last_colour = self.colour
            self.mcts_engine.colour = self.colour
            self.threat_detector.colour = self.colour
            self.pattern_recognizer.colour = self.colour

            # 关键：清空树/TT，避免旧视角统计污染
            self.mcts_engine.root_node = None
            self.mcts_engine.tt.clear()

        try:
            # 0. 时间安全检查
            remaining_time = self.time_manager.get_remaining_time()
            if remaining_time < 0.02:  # 少于50ms
                return self._get_quick_move(board)
            
            # 1. 处理对手的交换（如果对手在turn==2交换了，游戏引擎会自动改变我们的颜色）
            # 但我们需要更新has_swapped状态
            if opp_move and opp_move.is_swap():
                self.has_swapped = True
                # 游戏引擎已经改变了我们的颜色，但我们需要更新所有模块
                self.mcts_engine.colour = self.colour
                self.threat_detector.colour = self.colour
                self.pattern_recognizer.colour = self.colour
                # 不再手动reset
            
            # 2. 交换决策（第二回合，且对手没有交换）
            if turn == 2 and not self.has_swapped and (not opp_move or not opp_move.is_swap()):
                if self._should_swap(board, opp_move):
                    self.has_swapped = True
                    logger.info("Agent decided to SWAP")
                    return Move(-1, -1)
                # 决定不 swap：不要 return，继续往下走威胁检测 + MCTS

            # 3. 战术威胁检测（最高优先级，v2）
            my_colour = self.colour
            opp_colour = self.opp_colour()

            # 3.1 我方一手必胜
            my_wins = self.threat_detector.immediate_win_moves(board, my_colour)
            if my_wins:
                x, y = my_wins[0]
                return Move(x, y)

            # 3.2 对手一手必胜点 -> 必须挡
            opp_wins = self.threat_detector.immediate_win_moves(board, opp_colour)
            if opp_wins:
                x, y = opp_wins[0]
                return Move(x, y)

            # 3.3 桥战术响应：对手点桥的一侧，我方补另一侧（强制）
            bridge_blocks = self.threat_detector.bridge_responses(board, my_colour)
            if bridge_blocks:
                # 这里可能有多个，选一个即可（也可按 pattern prior 排序）
                x, y = bridge_blocks[0]
                return Move(x, y)

            # 3.4 2-ply forcing / double threat（候选用 pattern topK 限制）
            # 候选生成：用 pattern_recognizer 的简单模式点 + 中心点兜底
            candidates = []
            try:
                pats = self.pattern_recognizer.detect_simple_patterns(board, my_colour)
                if pats:
                    pats.sort(key=lambda p: p[2], reverse=True)
                    candidates = [(x, y) for x, y, _ in pats[:18]]
            except:
                candidates = []

            forcing = self.threat_detector.forcing_win_moves_2ply(
                board, my_colour, candidates=candidates, max_candidates=18
            )
            if forcing:
                x, y = forcing[0]
                return Move(x, y)

            # 3.5 Cut threat：显著增加对手最短路代价的点（很像“切断”）
            cut_list = self.threat_detector.cut_moves(board, my_colour, top_k=8)
            if cut_list:
                x, y = cut_list[0]
                return Move(x, y)

            # 4. 时间分配
            total_turns_remaining = self._estimate_remaining_turns(board)
            time_budget = self.time_manager.allocate_time(
                board, 
                remaining_time,
                total_turns_remaining,
                self.threat_detector,
                self.colour
            )
            
            # 5. MCTS搜索
            try:
                move = self.mcts_engine.search(board, time_budget)
                
                # 三重验证走法合法性（确保万无一失）
                if not move:
                    logger.warning("MCTS returned None, using fallback")
                    return self._get_fallback_move(board)
                
                # 验证1：基本合法性
                if not self._is_valid_move(move, board, turn):
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
        finally:
            self.time_manager.end_move()

    def _should_swap(self, board: Board, opp_move: Move | None) -> bool:
        if self.has_swapped:
            return False

        try:
            red_cost = self.connectivity_evaluator.shortest_path_cost(board, Colour.RED)
            blue_cost = self.connectivity_evaluator.shortest_path_cost(board, Colour.BLUE)

            if self.colour == Colour.RED:
                my_advantage = blue_cost - red_cost
            else:
                my_advantage = red_cost - blue_cost

            return my_advantage < 0
        except Exception as e:
            logger.error(f"Error in swap decision: {e}")
            return False

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
        时间过少时的快速走法（v2）
        优先级：我方一手赢 > 必防 > 桥响应 > 第一个合法点
        """
        try:
            my_colour = self.colour
            opp_colour = self.opp_colour()

            # 1) 我方一手必胜
            my_wins = self.threat_detector.immediate_win_moves(board, my_colour)
            if my_wins:
                x, y = my_wins[0]
                m = Move(x, y)
                if self._is_valid_move(m, board):
                    return m

            # 2) 必防
            opp_wins = self.threat_detector.immediate_win_moves(board, opp_colour)
            if opp_wins:
                x, y = opp_wins[0]
                m = Move(x, y)
                if self._is_valid_move(m, board):
                    return m

            # 3) 桥响应
            bridge_blocks = self.threat_detector.bridge_responses(board, my_colour)
            if bridge_blocks:
                x, y = bridge_blocks[0]
                m = Move(x, y)
                if self._is_valid_move(m, board):
                    return m

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

    def _is_valid_move(self, move: Move, board: Board, turn: int | None = None) -> bool:
        if turn is None:
            turn = self.turn_count
        if move.x == -1 and move.y == -1:
            return turn == 2 and not self.has_swapped
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

