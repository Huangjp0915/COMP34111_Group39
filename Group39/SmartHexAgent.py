"""
Group39 Hex AI Agent - 主代理类
EH-v7.4 - Logic Fixes & Debug

修复日志：
1. [CRITICAL] 移除了 __init__ 中的 start_timer，防止计入对手时间。
2. 在 make_move 中正确管理 start_timer / stop_timer 的生命周期。
3. 集成了文件日志调试功能。
"""

import copy
import random
import math
from typing import Optional, List, Tuple

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from .mcts_engine import MCTSEngine
from .threat_detector import ThreatDetector
from .connectivity_evaluator import ConnectivityEvaluator
from .pattern_recognizer import PatternRecognizer
from .time_manager import TimeManager
from .utils import logger

class SmartHexAgent(AgentBase):
    """
    智能 Hex 代理 (MoHex 架构版)。
    """
    def __init__(self, colour: Colour):
        super().__init__(colour)
        
        # 1. 初始化核心模块
        self.connectivity_evaluator = ConnectivityEvaluator()
        self.pattern_recognizer = PatternRecognizer(colour)
        self.threat_detector = ThreatDetector(colour)
        self.mcts_engine = MCTSEngine(colour)
        
        # 注入依赖
        self.mcts_engine.connectivity_evaluator = self.connectivity_evaluator
        self.mcts_engine.pattern_recognizer = self.pattern_recognizer
        
        # 时间管理
        self.time_manager = TimeManager(self.colour)
        # [修复] 初始化时不再开启计时器！
        
        self.last_move = None
        self.turn_count = 0
        self.has_swapped = False

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        主决策入口
        """
        self.turn_count = turn
        
        # [计时开始] 仅当轮到我方行动时，才开始计时
        self.time_manager.start_turn_timer()
        
        # 强制同步颜色
        self._ensure_colour_sync()

        try:
            # --- 0. 基础状态维护 ---
            remaining_time = self.time_manager.get_remaining_time()
            
            # 处理对手的 Swap 动作
            if opp_move and opp_move.is_swap():
                self.has_swapped = True
                self.mcts_engine.reset_tree() 

            # 时间极少时的快速响应 (例如 < 0.5s)
            if remaining_time < 0.5:
                return self._get_quick_move(board)

            # --- 1. 开局阶段 ---
            if turn == 1:
                return self._choose_balanced_opening(board)

            if turn == 2 and not self.has_swapped and (not opp_move or not opp_move.is_swap()):
                swap = self._decide_swap(board, opp_move)
                if swap.x == -1: 
                    self.has_swapped = True
                    return swap

            # --- 2. 战术必应阶段 ---
            threats = self.threat_detector.detect_immediate_threats(board, self.opp_colour())
            for x, y, type_ in threats:
                if type_ == "WIN": return Move(x, y)
            for x, y, type_ in threats:
                if type_ == "LOSE": return Move(x, y)

            priority_move = self._check_priority_moves(board)
            if priority_move: return priority_move

            # --- 3. 战略防守阶段 ---
            defensive = self._choose_defensive_block(board)
            if defensive:
                logger.info(f"Strategic Block Triggered at {defensive}")
                self.last_move = defensive
                self._debug_visualize_weights(board, opp_move, defensive)
                return defensive

            # --- 4. MCTS 搜索阶段 ---
            rem_turns = self._estimate_remaining_turns(board)
            budget = self.time_manager.allocate_time(board, remaining_time, rem_turns, 
                                                    self.threat_detector, self.opp_colour())
            
            if opp_move and not opp_move.is_swap():
                self.mcts_engine.last_move = opp_move
            else:
                self.mcts_engine.last_move = None
                self.mcts_engine.reset_tree()
            
            best_move = self.mcts_engine.search(board, budget)

            # --- 5. 后处理与验证阶段 ---
            final_move = self._postprocess_mcts_move(board, best_move)
            
            if not self._is_valid_move(final_move, board):
                logger.error(f"Invalid move {final_move} generated, falling back.")
                final_move = self._get_fallback_move(board)

            self.last_move = final_move
            
            # [DEBUG] 输出权重分布到文件
            if final_move.x != -1:
                self._debug_visualize_weights(board, opp_move, final_move)
            else:
                logger.debug(f"\n[TURN {turn} DEBUG] Agent chose to SWAP.")

            return final_move

        except Exception as e:
            logger.error(f"Critical error in make_move: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_move(board)
        
        finally:
            # [计时结束] 停止计时并累加本回合耗时
            self.time_manager.stop_turn_timer()

    # ==========================================
    # 辅助与同步
    # ==========================================

    def _ensure_colour_sync(self):
        """确保所有子模块的颜色与 Agent 当前颜色一致"""
        if self.pattern_recognizer.colour != self.colour:
            logger.info(f"Syncing colour from {self.pattern_recognizer.colour} to {self.colour}")
            self.pattern_recognizer.colour = self.colour
            self.threat_detector.colour = self.colour
            self.mcts_engine.colour = self.colour
            self.pattern_recognizer._cache.clear()
            self.mcts_engine.reset_tree()

    # ==========================================
    # 策略子模块
    # ==========================================

    def _choose_balanced_opening(self, board: Board) -> Move:
        size = board.size
        candidates = [
            ((1, size//2), 5),
            ((2, size//2 - 1), 4),
            ((1, size//2 + 1), 4),
            ((0, size - 3), 3),
            ((size//2, 0), 2)
        ]
        
        valid_points = []
        valid_weights = []
        
        for (r, c), w in candidates:
            # 适配坐标系: Red(Row), Blue(Col)
            if self.colour == Colour.RED:
                x, y = r, c 
            else:
                x, y = c, r 
            
            if 0 <= x < size and 0 <= y < size and board.tiles[x][y].colour is None:
                valid_points.append(Move(x, y))
                valid_weights.append(w)
        
        if valid_points:
            return random.choices(valid_points, weights=valid_weights, k=1)[0]
        return self._get_first_valid_move(board)

    def _decide_swap(self, board: Board, opp_move: Move) -> Move:
        if self.has_swapped: return self._get_fallback_move(board)
        
        my_c = self.colour
        opp_c = self.opp_colour()
        
        best_stay_adv = -999.0
        cands = self._generate_local_candidates(board, my_c, limit=15)
        for x, y in cands:
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, my_c)
            mc = self.connectivity_evaluator.shortest_path_cost(b_sim, my_c)
            oc = self.connectivity_evaluator.shortest_path_cost(b_sim, opp_c)
            adv = oc - mc
            if adv > best_stay_adv:
                best_stay_adv = adv
        
        b_swap = copy.deepcopy(board)
        b_swap.set_tile_colour(opp_move.x, opp_move.y, opp_c)
        
        mc_swap = self.connectivity_evaluator.shortest_path_cost(b_swap, opp_c)
        oc_swap = self.connectivity_evaluator.shortest_path_cost(b_swap, my_c)
        adv_swap = (oc_swap - mc_swap) - 0.7
        
        if adv_swap > best_stay_adv + 0.2:
            logger.info(f"Swap Decision: STAY={best_stay_adv:.2f}, SWAP={adv_swap:.2f} -> SWAP")
            return Move(-1, -1)
            
        return self._get_fallback_move(board)

    def _choose_defensive_block(self, board: Board) -> Optional[Move]:
        opp_c = self.opp_colour()
        my_c = self.colour
        
        base_opp_cost = self.connectivity_evaluator.shortest_path_cost(board, opp_c)
        
        if base_opp_cost > 3.5:
            return None 

        candidates = self._generate_local_candidates(board, opp_c, limit=30)
        best_block = None
        max_gain = 0.0
        
        for x, y in candidates:
            if self.pattern_recognizer._is_inferior(board, x, y, my_c):
                continue
                
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, my_c)
            
            new_opp_cost = self.connectivity_evaluator.shortest_path_cost(b_sim, opp_c)
            gain = new_opp_cost - base_opp_cost
            
            if gain > max_gain:
                max_gain = gain
                best_block = Move(x, y)
        
        if best_block and max_gain >= 0.5:
            return best_block
            
        return None

    def _postprocess_mcts_move(self, board: Board, move: Move) -> Move:
        if not move or (move.x == -1): return move
        
        if self.pattern_recognizer._is_inferior(board, move.x, move.y, self.colour):
            logger.warning(f"Postprocess: Override Inferior Move {move}")
            return self._find_best_alternative(board)
            
        return move

    def _find_best_alternative(self, board: Board) -> Move:
        best = self._get_first_valid_move(board)
        best_score = -999.0
        
        cands = self._generate_local_candidates(board, self.colour, limit=20)
        cands += self._generate_local_candidates(board, self.opp_colour(), limit=20)
        
        for x, y in set(cands):
            mv = Move(x, y)
            if not self._is_valid_move(mv, board): continue
            
            if self.pattern_recognizer._is_inferior(board, x, y, self.colour):
                continue
                
            s = self._score_move_light(board, mv)
            if s > best_score:
                best_score = s
                best = mv
        return best

    def _score_move_light(self, board: Board, move: Move) -> float:
        if move.x == -1: return -99.0
        
        my = self.colour
        opp = self.opp_colour()
        
        if self.pattern_recognizer._is_inferior(board, move.x, move.y, self.colour):
            return -100.0

        cur_my = self.connectivity_evaluator.shortest_path_cost(board, my)
        cur_opp = self.connectivity_evaluator.shortest_path_cost(board, opp)
        
        b2 = copy.deepcopy(board)
        b2.set_tile_colour(move.x, move.y, my)
        
        new_my = self.connectivity_evaluator.shortest_path_cost(b2, my)
        new_opp = self.connectivity_evaluator.shortest_path_cost(b2, opp)
        
        opp_gain = new_opp - cur_opp
        my_gain = cur_my - new_my
        
        gamma_score = self.pattern_recognizer.get_prior(board, move, my)
        
        return opp_gain + 0.7 * my_gain + 0.5 * gamma_score

    # ==========================================
    # 调试工具 (Debug Tools)
    # ==========================================

    def _debug_visualize_weights(self, board: Board, opp_move: Optional[Move], selected_move: Move):
        """
        [DEBUG] 将全盘权重分布格式化为字符串，并记录到日志文件 (DEBUG Level)。
        """
        lines = []
        lines.append(f"\n{'='*20} [TURN {self.turn_count} DEBUG] {'='*20}")
        lines.append(f"My Colour: {self.colour}, Opponent Last: {opp_move}")
        
        size = board.size
        empty_spots = []
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    empty_spots.append(Move(x, y))
        
        gammas = self.pattern_recognizer.get_all_gammas(board, empty_spots, opp_move)
        
        sel_gamma = 0.0
        if selected_move.x != -1:
            sel_gamma = gammas.get((selected_move.x, selected_move.y), 0.0)
            lines.append(f">> SELECTED MOVE: {selected_move} | Gamma: {sel_gamma:.4f}")
        else:
            lines.append(f">> SELECTED MOVE: SWAP")

        # Top 10
        sorted_spots = sorted(gammas.items(), key=lambda x: x[1], reverse=True)
        lines.append(f"\n--- Top 10 Candidates (By Gamma) ---")
        for i, ((x, y), g) in enumerate(sorted_spots[:10]):
            mark = " <== CHOSEN" if (x == selected_move.x and y == selected_move.y) else ""
            lines.append(f"#{i+1:2d}: ({x}, {y})  Val: {g:8.4f}{mark}")

        # Heatmap
        lines.append(f"\n--- Weight Heatmap (Grid View) ---")
        lines.append("     " + " ".join([f"{y:^6}" for y in range(size)]))
        
        for x in range(size):
            row_str = f"{x:2d} | "
            for y in range(size):
                tile = board.tiles[x][y]
                if tile.colour == Colour.RED:
                    val_str = "  R   "
                elif tile.colour == Colour.BLUE:
                    val_str = "  B   "
                else:
                    g = gammas.get((x, y), 0.0)
                    if g > 100:
                        val_str = f"*{g:4.0f}*"
                    elif g > 10:
                        val_str = f" {g:4.1f} "
                    elif g < 0.01:
                        val_str = "   .  "
                    else:
                        val_str = f" {g:4.1f} "
                row_str += val_str + "|"
            lines.append(row_str)
        lines.append("="*60 + "\n")
        
        logger.debug("\n".join(lines))

    # ==========================================
    # 基础工具
    # ==========================================

    def _check_priority_moves(self, board: Board) -> Optional[Move]:
        win_cands = self.threat_detector._get_candidates(board, self.colour)
        for x, y in win_cands:
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, self.colour)
            if b_sim.has_ended(self.colour): return Move(x, y)
            
        opp = self.opp_colour()
        lose_cands = self.threat_detector._get_candidates(board, opp)
        for x, y in lose_cands:
            if self.pattern_recognizer._is_inferior(board, x, y, self.colour):
                continue
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, opp)
            if b_sim.has_ended(opp): return Move(x, y)
        return None

    def _generate_local_candidates(self, board: Board, focus_colour: Colour, limit=40) -> List[Tuple[int, int]]:
        candidates = set()
        size = board.size
        
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == focus_colour:
                    from src.Tile import Tile
                    for i in range(Tile.NEIGHBOUR_COUNT):
                        nx, ny = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
                        if 0 <= nx < size and 0 <= ny < size and board.tiles[nx][ny].colour is None:
                            candidates.add((nx, ny))
        
        if focus_colour == Colour.RED:
            for y in range(size):
                if board.tiles[1][y].colour is None: candidates.add((1, y))
                if board.tiles[size-2][y].colour is None: candidates.add((size-2, y))
        else:
            for x in range(size):
                if board.tiles[x][1].colour is None: candidates.add((x, 1))
                if board.tiles[x][size-2].colour is None: candidates.add((x, size-2))

        return list(candidates)[:limit]

    def _get_quick_move(self, board: Board) -> Move:
        """时间不足时的快速走法"""
        # [日志] 记录快速走法被触发
        logger.debug("Quick move triggered!")
        
        threats = self.threat_detector.detect_immediate_threats(board, self.opp_colour())
        if threats: return Move(threats[0][0], threats[0][1])
        return self._get_first_valid_move(board)

    def _get_first_valid_move(self, board: Board) -> Move:
        size = board.size
        c = size // 2
        best_dist = 999
        best_move = Move(-1, -1)
        
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    dist = abs(x - c) + abs(y - c)
                    if dist < best_dist:
                        best_dist = dist
                        best_move = Move(x, y)
        return best_move
    
    def _get_fallback_move(self, board: Board) -> Move:
        try:
            if self.mcts_engine.root_node and self.mcts_engine.root_node.children:
                best = max(self.mcts_engine.root_node.children, key=lambda c: c.visits)
                if self._is_valid_move(best.move, board): return best.move
        except:
            pass
        return self._get_first_valid_move(board)

    def _is_valid_move(self, move: Move, board: Board) -> bool:
        if move.x == -1 and move.y == -1: return True
        return 0 <= move.x < board.size and 0 <= move.y < board.size and board.tiles[move.x][move.y].colour is None

    def _estimate_remaining_turns(self, board: Board) -> int:
        emp = sum(1 for x in range(board.size) for y in range(board.size) if board.tiles[x][y].colour is None)
        return max(1, emp // 2)

    def opp_colour(self) -> Colour:
        return Colour.opposite(self.colour)