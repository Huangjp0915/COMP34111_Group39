"""
Group39 Hex AI Agent - 主代理类
"""
# import copy
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
from typing import Optional

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

    def _find_winning_move_by_simulation(self, board: Board, colour: Colour) -> Optional[Move]:
        """
        快速检查：是否存在“一步即胜”的落子。
        优化点：
        - 不使用 deepcopy
        - 只在“高度相关”的候选空位里检查（速度更稳）
        """
        candidates = self._get_forced_check_candidates(board, colour)
        for x, y in candidates:
            if board.tiles[x][y].colour is not None:
                continue
            if self._try_place_and_check_win(board, x, y, colour):
                return Move(x, y)
        return None

    def _try_place_and_check_win(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """临时落子 -> has_ended -> 还原（避免 deepcopy）"""
        original = board.tiles[x][y].colour
        board.tiles[x][y].colour = colour
        try:
            return board.has_ended(colour)
        finally:
            board.tiles[x][y].colour = original

    def _generate_candidates(self, board: Board) -> list[tuple[int, int]]:
        """
        候选点生成（用于“约束输出/兜底替换”）
        思路：
        - 已落子邻域空位（最重要）
        - 中心带（开局避免贴边）
        - 可控上限K
        """
        from src.Tile import Tile
        size = board.size
        cand = set()

        # 统计空位数，判早期
        empties = 0
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    empties += 1

        # 1) 所有已落子周围的空位（双方都算）
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    continue
                for i in range(Tile.NEIGHBOUR_COUNT):
                    nx = x + Tile.I_DISPLACEMENTS[i]
                    ny = y + Tile.J_DISPLACEMENTS[i]
                    if 0 <= nx < size and 0 <= ny < size and board.tiles[nx][ny].colour is None:
                        cand.add((nx, ny))

        # 2) 早期：加中心菱形（防贴边）
        if empties >= size * size - 10:
            cx, cy = size // 2, size // 2
            radius = 3
            for x in range(max(0, cx - radius), min(size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(size, cy + radius + 1)):
                    if board.tiles[x][y].colour is None:
                        cand.add((x, y))

        # 3) 若候选太少：再补一圈中心
        if len(cand) < 12:
            cx, cy = size // 2, size // 2
            radius = 4
            for x in range(max(0, cx - radius), min(size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(size, cy + radius + 1)):
                    if board.tiles[x][y].colour is None:
                        cand.add((x, y))

        cand_list = list(cand)

        K = 48
        if len(cand_list) > K:
            # 收集所有已落子（只算一次）
            stones = []
            for x in range(size):
                for y in range(size):
                    if board.tiles[x][y].colour is not None:
                        stones.append((x, y))

            cx, cy = size // 2, size // 2

            def min_dist_to_stone(px, py):
                if not stones:
                    return 999  # 极端：棋盘空
                return min(abs(px - sx) + abs(py - sy) for sx, sy in stones)

            # 先靠近棋形（min_dist 小），再偏中心一点点（center_dist 小）
            cand_list.sort(key=lambda p: (min_dist_to_stone(p[0], p[1]), abs(p[0] - cx) + abs(p[1] - cy)))
            cand_list = cand_list[:K]

        return cand_list

    def _candidate_score(self, board: Board, x: int, y: int, before_my: float, before_opp: float) -> float:
        """
        候选点轻量打分：越大越好
        - 偏中心（温和）
        - 落子后我方路径成本下降（更关键）
        - 同时让对手成本上升（轻微）
        """
        size = board.size
        cx, cy = size // 2, size // 2
        center_dist = abs(x - cx) + abs(y - cy)

        my = self.colour
        opp = self.opp_colour()

        original = board.tiles[x][y].colour
        board.tiles[x][y].colour = my
        try:
            after_my = self.connectivity_evaluator.shortest_path_cost(board, my)
            after_opp = self.connectivity_evaluator.shortest_path_cost(board, opp)
        finally:
            board.tiles[x][y].colour = original

        def safe_improve(b, a):
            if b == float("inf") or a == float("inf"):
                return 0.0
            return b - a

        my_improve = safe_improve(before_my, after_my)
        opp_worsen = safe_improve(after_opp, before_opp)  # after_opp - before_opp

        center_bonus = max(0.0, 1.0 - center_dist / max(1, size))

        return 1.25 * my_improve + 0.40 * opp_worsen + 0.25 * center_bonus

    def _best_candidate_move(self, board: Board, candidates: list[tuple[int, int]]) -> Optional[Move]:
        before_my = self.connectivity_evaluator.shortest_path_cost(board, self.colour)
        before_opp = self.connectivity_evaluator.shortest_path_cost(board, self.opp_colour())

        best = None
        best_s = float("-inf")
        for x, y in candidates:
            if board.tiles[x][y].colour is not None:
                continue
            s = self._candidate_score(board, x, y, before_my, before_opp)
            if (s > best_s) or (s == best_s and (best is None or (x, y) < best)):
                best_s = s
                best = (x, y)
        return Move(best[0], best[1]) if best else None

    def _get_forced_check_candidates(self, board: Board, colour: Colour) -> list[tuple[int, int]]:
        """
        给“一步胜/一步防”专用的候选点集合：
        - 所有空位中，优先取：靠近己方棋/靠近对手棋/靠近目标边 的点
        目的：把 121 个点缩到常见 20~50 个以内。
        """
        from src.Tile import Tile

        size = board.size
        cand = set()

        # 1) 所有已落子周围的空位（双方都算）
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    continue
                for i in range(Tile.NEIGHBOUR_COUNT):
                    nx = x + Tile.I_DISPLACEMENTS[i]
                    ny = y + Tile.J_DISPLACEMENTS[i]
                    if 0 <= nx < size and 0 <= ny < size and board.tiles[nx][ny].colour is None:
                        cand.add((nx, ny))

        # 2) 靠近自己要连接的两条边（Red: top/bottom; Blue: left/right）
        edge_band = 2  # 边缘附近的“带宽”，越大越慢；2 一般够用
        if colour == Colour.RED:
            for y in range(size):
                for x in range(0, min(edge_band, size)):
                    if board.tiles[x][y].colour is None:
                        cand.add((x, y))
                for x in range(max(0, size - edge_band), size):
                    if board.tiles[x][y].colour is None:
                        cand.add((x, y))
        else:
            for x in range(size):
                for y in range(0, min(edge_band, size)):
                    if board.tiles[x][y].colour is None:
                        cand.add((x, y))
                for y in range(max(0, size - edge_band), size):
                    if board.tiles[x][y].colour is None:
                        cand.add((x, y))

        # 3) 如果棋盘几乎空（前几手），补一点中心附近，避免“候选集太偏边”
        empties = 0
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    empties += 1
        if empties >= size * size - 6:
            cx = size // 2
            cy = size // 2
            radius = 2
            for x in range(max(0, cx - radius), min(size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(size, cy + radius + 1)):
                    if board.tiles[x][y].colour is None:
                        cand.add((x, y))

        return list(cand)

    def _check_priority_moves(self, board: Board) -> Optional[Move]:
        """
        显式规则层：
        1. 先看我方有没有“一步必杀”
        2. 再看对手有没有“一步必胜”（需要必防）
        都没有则返回 None
        """
        # 1) 我方必杀
        win_move = self._find_winning_move_by_simulation(board, self.colour)
        if win_move is not None:
            return win_move

        # 2) 对方必胜 => 我方必防
        opp = self.opp_colour()
        block_move = self._find_winning_move_by_simulation(board, opp)
        if block_move is not None:
            return block_move

        return None

    def _is_high_tension(self, board: Board) -> bool:
        opp = self.opp_colour()
        opp_cost = self.connectivity_evaluator.shortest_path_cost(board, opp)
        empties = sum(
            1 for x in range(board.size) for y in range(board.size)
            if board.tiles[x][y].colour is None
        )
        return (opp_cost != float("inf") and opp_cost <= 3) or (empties <= board.size * 2)

    def _gives_opp_one_move_win(self, board, my_move) -> bool:
        if my_move is None or my_move.is_swap():
            return False
        if not self._is_valid_move(my_move, board):
            return True
        opp = self.opp_colour()
        ox = board.tiles[my_move.x][my_move.y].colour
        board.tiles[my_move.x][my_move.y].colour = self.colour
        try:
            return self._find_winning_move_by_simulation(board, opp) is not None
        finally:
            board.tiles[my_move.x][my_move.y].colour = ox

    def _is_board_empty(self, board: Board) -> bool:
        """检查棋盘是否完全为空,用于规范先手位置"""
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is not None:
                    return False
        return True

    def _is_my_first_move(self, board: Board) -> bool:
        """判断这是不是我在本局中的第一颗棋子。无论我是先手还是后手/是否发生过交换，都只看棋盘上有没有属于我的棋子。
        """
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == self.colour:
                    return False
        return True

    def _sync_modules_colour_if_needed(self):
        """
        确保在发生 swap（无论是谁 swap）后，
        所有子模块的 colour 与 self.colour 保持一致，并清理 MCTS 根状态，
        这样不会依赖“swap是谁做的”，更稳妥些（吧？）
        """
        if getattr(self.mcts_engine, "colour", None) != self.colour:
            self.mcts_engine.colour = self.colour
            self.threat_detector.colour = self.colour
            self.pattern_recognizer.colour = self.colour

            # 换色后：MCTS 根与 last_move/board_hash 都必须清空
            self.mcts_engine.root_node = None
            self.mcts_engine.last_move = None
            self.mcts_engine.last_board_hash = None

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
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
            if remaining_time < 0.05:
                return self._get_quick_move(board)

            self._sync_modules_colour_if_needed()  #触发一下 swap 同步函数保险机制

            # 1. 对手在 turn==2 选择 swap：这里只做状态标记。
            # 模块 colour 同步与 MCTS 清理由 _sync_modules_colour_if_needed() 统一处理。
            if opp_move and opp_move.is_swap():
                self.has_swapped = True
            
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

            # 3.2 规则层“必杀必防”兜底检查（独立于 ThreatDetector）
            priority_move = self._check_priority_moves(board)
            if priority_move is not None:
                return priority_move

            # 3.3 如果这是我在本局的第一颗棋子
            if self._is_my_first_move(board):
                if turn == 1 and self._is_board_empty(board):
                    move = self._get_balanced_first_move(board)  # 避免被swap白嫖
                else:
                    move = self._get_center_biased_opening_reply(board)
                if self._is_valid_move(move, board):
                    self.last_move = move
                    return move
            
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

                # 先做 candidate 约束（把 MCTS 输出拉回“可控集合”）
                candidates = self._generate_candidates(board)
                if candidates and move and move.x != -1 and move.y != -1:
                    cand_set = set(candidates)
                    if (move.x, move.y) not in cand_set:
                        alt = self._best_candidate_move(board, candidates)
                        if alt and self._is_valid_move(alt, board):
                            move = alt

                # 再做 tension gating（对最终 move 做检查）
                if candidates is None:
                    candidates = []

                if self._is_high_tension(board) and remaining_time > 0.10:
                    if self._gives_opp_one_move_win(board, move):
                        logger.warning("Move gives opponent a one-move win; choosing best safe candidate")

                        safe_xy = []
                        for x, y in candidates:
                            m = Move(x, y)
                            if self._is_valid_move(m, board) and (not self._gives_opp_one_move_win(board, m)):
                                safe_xy.append((x, y))

                        if safe_xy:
                            alt = self._best_candidate_move(board, safe_xy)  # ✅ 关键：选 best，而不是选第一个
                            if alt and self._is_valid_move(alt, board):
                                move = alt
                            else:
                                move = self._get_fallback_move(board)
                        else:
                            move = self._get_fallback_move(board)
                
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
    
    def _opposite_colour_local(self, c: Colour) -> Colour:
        return Colour.BLUE if c == Colour.RED else Colour.RED

    def _opening_strength_score(self, board: Board, x: int, y: int, opener_colour: Colour) -> float:
        """
        评估“某个开局点对开局方(opener_colour)有多强” -> 0~1（大致）
        越大：越值得后手 swap。
        只做轻量计算：中心性 + 连通成本差（优势）+ 自己接近连通程度
        """
        if not (0 <= x < board.size and 0 <= y < board.size):
            return 0.0
        if board.tiles[x][y].colour is not None:
            return 0.0

        size = board.size
        cx, cy = size // 2, size // 2
        dist = abs(x - cx) + abs(y - cy)

        other_colour = self._opposite_colour_local(opener_colour)

        original = board.tiles[x][y].colour
        board.tiles[x][y].colour = opener_colour
        try:
            opener_cost = self.connectivity_evaluator.shortest_path_cost(board, opener_colour)
            other_cost = self.connectivity_evaluator.shortest_path_cost(board, other_colour)
        finally:
            board.tiles[x][y].colour = original

        def norm_cost(c: float) -> float:
            if c == float("inf"):
                return 1.0
            return max(0.0, min(1.0, c / max(1, size)))

        opener_n = norm_cost(opener_cost)
        other_n = norm_cost(other_cost)

        # opener 越接近连通越强
        closeness = 1.0 - opener_n  # 0~1
        # opener 相对优势：other_cost - opener_cost 越大越强
        advantage = max(-1.0, min(1.0, other_n - opener_n))  # -1~1（一般 0~1）
        # 越靠中心越强（简单经验）
        center_bonus = max(0.0, 1.0 - dist / max(1, size))  # 约 0~1

        # 可解释的线性组合（你后面想调参很方便）
        strength = 0.45 * advantage + 0.35 * closeness + 0.20 * center_bonus
        return max(0.0, min(1.0, strength))

    def _should_swap_by_opening_strength(self, board: Board, opp_move: Move) -> bool:
        """
        后手只看“开局强度”决定要不要 swap。
        """
        if not (0 <= opp_move.x < board.size and 0 <= opp_move.y < board.size):
            return False

        size = board.size
        opener_colour = board.tiles[opp_move.x][opp_move.y].colour
        if opener_colour is None:
            opener_colour = self.opp_colour()  # 极端兜底
        s = self._opening_strength_score(board, opp_move.x, opp_move.y, opener_colour)

        # 阈值（11x11 常用一套；如果你们不是11，可略调）
        STRONG = 0.34   # >= 强开局：swap
        WEAK   = 0.20   # <= 弱/平衡：不swap

        if s >= STRONG:
            return True
        if s <= WEAK:
            return False

        # 灰区：用中心距离做个小 tie-break（避免抖动）
        cx, cy = size // 2, size // 2
        dist = abs(opp_move.x - cx) + abs(opp_move.y - cy)
        return dist <= 2

    def _decide_swap(self, board: Board, opp_move: Optional[Move]) -> Move:
        """
        第二手只看开局强度
        """
        if self.has_swapped:
            return self._get_first_valid_move(board)
        if opp_move is None or opp_move.is_swap():
            return self._get_first_valid_move(board)

        if self._should_swap_by_opening_strength(board, opp_move):
            self.has_swapped = True
            logger.info("Agent decided to SWAP (opening strength heuristic)")
            return Move(-1, -1)

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

    def _get_balanced_first_move(self, board: Board) -> Move:
        """
        有 swap 规则时的先手第一步：追求“平衡”，避免对手爽swap。
        - 仍然偏中心（太边会弱）
        - 但避免“正中心/紧贴中心”这种明显强开局
        - 用一个轻量指标：落子后 red_cost 与 blue_cost 的差距尽量小（更“平衡”）
        """
        size = board.size
        cx = size // 2
        cy = size // 2

        best = None
        best_score = float("inf")

        radius = 4  # 兼顾速度

        for x in range(max(0, cx - radius), min(size, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(size, cy + radius + 1)):
                if board.tiles[x][y].colour is not None:
                    continue

                center_dist = abs(x - cx) + abs(y - cy)

                # 避免“太明显强”的点：正中心或紧贴中心
                strong_penalty = 0.0
                if center_dist <= 1:
                    strong_penalty = 2.0

                original = board.tiles[x][y].colour
                board.tiles[x][y].colour = self.colour
                try:
                    red_cost = self.connectivity_evaluator.shortest_path_cost(board, Colour.RED)
                    blue_cost = self.connectivity_evaluator.shortest_path_cost(board, Colour.BLUE)
                finally:
                    board.tiles[x][y].colour = original

                # 处理 inf
                if red_cost == float("inf") or blue_cost == float("inf"):
                    balance = 10.0
                else:
                    balance = abs(red_cost - blue_cost)

                # score 越小越好：更平衡 + 不太离谱地偏中心 + 不给对手明显swap理由
                score = 3.0 * balance + 0.20 * center_dist + strong_penalty

                strength = self._opening_strength_score(board, x, y, self.colour)
                score += 2.0 * max(0.0, strength - 0.30)

                if score < best_score:
                    best_score = score
                    best = (x, y)

        if best is None:
            return Move(cx, cy)  # 极端兜底
        return Move(best[0], best[1])

    def _get_center_biased_opening_reply(self, board: Board) -> Move:
        """
        用于“我在本局第一颗棋子”的开局/换色后第一手：
        - 明确偏好中心区域
        - 同时轻微考虑连通成本（别光图中心）
        """
        size = board.size
        cx = size // 2
        cy = size // 2

        best = None
        best_score = float("inf")

        # 只扫一个中心菱形区域（半径 3），非常快
        radius = 3
        for x in range(max(0, cx - radius), min(size, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(size, cy + radius + 1)):
                if board.tiles[x][y].colour is not None:
                    continue

                # score 越小越好：中心距离 + 轻量连通启发
                center_dist = abs(x - cx) + abs(y - cy)

                # 临时落子评估：我方路径成本下降越多越好
                before = self.connectivity_evaluator.shortest_path_cost(board, self.colour)
                original = board.tiles[x][y].colour
                board.tiles[x][y].colour = self.colour
                try:
                    after = self.connectivity_evaluator.shortest_path_cost(board, self.colour)
                finally:
                    board.tiles[x][y].colour = original

                improve = 0.0
                if before != float("inf") and after != float("inf"):
                    improve = before - after

                score = center_dist - 0.6 * improve  # 0.6 是温和权重（先别太激进）
                if score < best_score:
                    best_score = score
                    best = (x, y)

        if best is not None:
            return Move(best[0], best[1])

        # 兜底：用你已有的中心距离排序
        return self._get_first_valid_move(board)

    def _get_first_valid_move(self, board: Board) -> Move:
        """
        返回一个“相对合理”的合法走法：
        - 开局/兜底时，优先选择靠近棋盘中心的空位
        - 如果没有空位（理论上不会发生），返回交换
        """
        size = board.size
        center = size // 2
        candidates = []

        # 收集所有空位，并计算到中心的曼哈顿距离
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    dist = abs(x - center) + abs(y - center)
                    candidates.append((dist, x, y))

        if not candidates:
            return Move(-1, -1)

        # 按距离升序排序，取距离最小的那个
        candidates.sort(key=lambda t: t[0])
        _, best_x, best_y = candidates[0]
        return Move(best_x, best_y)
    
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
