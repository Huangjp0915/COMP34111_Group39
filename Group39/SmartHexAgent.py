"""
Group39 Hex AI Agent - 主代理类
EH-v7.0 - MoHex Architecture Integrated (Full Version)

集成特性：
1. MoHex Style Patterns: 利用 Gamma 权重识别 Local/Global 模式。
2. Dijkstra Evaluator: 基于势能场的全局距离评估。
3. Consistent Scoring: MCTS 与后处理使用统一的战术评价标准。
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
        # Evaluator: 提供 Dijkstra + VC 距离评估
        self.connectivity_evaluator = ConnectivityEvaluator()

        # Pattern: 提供 MoHex 风格的 Gamma 战术权重
        self.pattern_recognizer = PatternRecognizer(colour)

        # Threat: 提供必胜/必败的几何检测
        self.threat_detector = ThreatDetector(colour)

        # MCTS: 搜索引擎 (Gamma -> Prior)
        self.mcts_engine = MCTSEngine(colour)

        # [关键] 依赖注入
        self.mcts_engine.connectivity_evaluator = self.connectivity_evaluator
        self.mcts_engine.pattern_recognizer = self.pattern_recognizer

        # [修改] 时间管理 - 传入颜色，用于静态变量区分红/蓝时钟
        self.time_manager = TimeManager(self.colour)

        # 状态跟踪
        self.last_move = None
        self.turn_count = 0
        self.has_swapped = False

    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        主决策入口
        """
        self.time_manager.start_turn_timer()
        self.turn_count = turn

        try:
            # --- 0. 基础状态维护 ---
            # 检查剩余时间
            remaining_time = self.time_manager.get_remaining_time()
            if remaining_time < 0.05:
                return self._get_quick_move(board)

            # 处理对手的 Swap 动作
            if opp_move and opp_move.is_swap():
                self.has_swapped = True
                self._update_modules_colour(self.colour) # 重新同步颜色状态
                self.mcts_engine.reset_tree() # 局势突变，旧树失效

            # --- 1. 开局阶段 ---
            # Turn 1: 使用开局库
            if turn == 1:
                return self._choose_balanced_opening(board)

            # Turn 2: Swap 决策 (基于 Dijkstra Advantage)
            if turn == 2 and not self.has_swapped and (not opp_move or not opp_move.is_swap()):
                swap = self._decide_swap(board, opp_move)
                if swap.x == -1:
                    self.has_swapped = True
                    # 注意: 游戏引擎会在收到 swap 后自动处理颜色变更
                    return swap

            # --- 2. 战术必应阶段 (Highest Priority) ---
            # 威胁检测：直接赢 > 必须防
            threats = self.threat_detector.detect_immediate_threats(board, self.opp_colour())

            for x, y, type_ in threats:
                if type_ == "WIN": return Move(x, y)
            for x, y, type_ in threats:
                if type_ == "LOSE": return Move(x, y)

            # 规则兜底 (Double Check)
            priority_move = self._check_priority_moves(board)
            if priority_move: return priority_move

            # --- 3. 战略防守阶段 (Strategic Defense) ---
            # 利用 Dijkstra 的势能场判断对手是否形成了强力 VC
            # 如果是，尝试寻找能最大程度增加对手 Cost 的"镇头"或"肩冲"点
            defensive = self._choose_defensive_block(board)
            if defensive:
                logger.info(f"Strategic Block Triggered at {defensive}")
                return defensive

            # --- 4. MCTS 搜索阶段 (Main Search) ---
            # 估算剩余回合
            rem_turns = self._estimate_remaining_turns(board)
            # 分配时间预算
            budget = self.time_manager.allocate_time(board, remaining_time, rem_turns,
                                                    self.threat_detector, self.opp_colour())

            # 更新 MCTS 状态 (传入对手上一手，用于激活 Local Patterns)
            if opp_move and not opp_move.is_swap():
                self.mcts_engine.last_move = opp_move
            else:
                self.mcts_engine.last_move = None
                self.mcts_engine.reset_tree()

            # 执行搜索
            best_move = self.mcts_engine.search(board, budget)

            # --- 5. 后处理与验证阶段 ---
            # 利用 PatternRecognizer 的 Inferior Check 进行二次过滤
            final_move = self._postprocess_mcts_move(board, best_move)

            # 最终合法性检查 (防止任何可能的异常)
            if not self._is_valid_move(final_move, board):
                logger.error(f"Invalid move {final_move} generated, falling back.")
                return self._get_fallback_move(board)

            self.last_move = final_move
            return final_move

        except Exception as e:
            logger.error(f"Critical error in make_move: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_move(board)

        # 无论如何（正常返回或异常），在函数退出前停止计时
        finally:
            self.time_manager.stop_turn_timer()

    # ==========================================
    # 策略子模块
    # ==========================================

    def _choose_balanced_opening(self, board: Board) -> Move:
        """
        开局策略：选择经典的平衡点 (a8, b7 等)。
        使用加权随机以避免被对手背谱针对。
        """
        size = board.size
        # 候选列表 [(row, col), weight] (基于 11x11)
        # 注意：这里使用逻辑坐标 (Row, Col)，后面会根据颜色转换
        # Red 连接 Top-Bottom (Rows), Blue 连接 Left-Right (Cols)
        candidates = [
            ((1, size//2), 5),      # 防守型
            ((2, size//2 - 1), 4),  # 平衡型
            ((1, size//2 + 1), 4),  # 平衡型
            ((0, size - 3), 3),     # 诱导 Swap
            ((size//2, 0), 2)       # 边路开局
        ]
        
        valid_points = []
        valid_weights = []
        
        for (r, c), w in candidates:
            # 适配坐标系
            if self.colour == Colour.RED:
                x, y = r, c # x是行
            else:
                x, y = c, r # y是行
            
            if 0 <= x < size and 0 <= y < size and board.tiles[x][y].colour is None:
                valid_points.append(Move(x, y))
                valid_weights.append(w)
        
        if valid_points:
            return random.choices(valid_points, weights=valid_weights, k=1)[0]
        
        # 兜底：中心空位
        return self._get_first_valid_move(board)

    def _decide_swap(self, board: Board, opp_move: Move) -> Move:
        """
        Swap 决策：利用 Dijkstra Evaluator 计算双方势能差 (Advantage)。
        """
        if self.has_swapped: return self._get_fallback_move(board)
        
        my_c = self.colour
        opp_c = self.opp_colour()
        
        # 1. 评估 STAY (我不换，我继续下)
        best_stay_adv = -999.0
        # 快速扫描几个好点
        cands = self._generate_local_candidates(board, my_c, limit=15)
        for x, y in cands:
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, my_c)
            
            mc = self.connectivity_evaluator.shortest_path_cost(b_sim, my_c)
            oc = self.connectivity_evaluator.shortest_path_cost(b_sim, opp_c)
            # Adv = 敌方代价 - 我方代价 (我方代价越小越好)
            adv = oc - mc
            if adv > best_stay_adv:
                best_stay_adv = adv
        
        # 2. 评估 SWAP (我换成对手颜色)
        b_swap = copy.deepcopy(board)
        # 对手那一子归我 (颜色变为 opp_c)
        b_swap.set_tile_colour(opp_move.x, opp_move.y, opp_c)
        
        # 我现在是 opp_c，轮到对手 (my_c) 下
        # 评估静态盘面优势
        mc_swap = self.connectivity_evaluator.shortest_path_cost(b_swap, opp_c) # 我是OppC
        oc_swap = self.connectivity_evaluator.shortest_path_cost(b_swap, my_c)  # 敌是MyC
        
        # 减去先手惩罚 (对手有一手棋权)
        adv_swap = (oc_swap - mc_swap) - 0.7
        
        # 决策：如果 Swap 优势显著
        if adv_swap > best_stay_adv + 0.2:
            logger.info(f"Swap Decision: STAY={best_stay_adv:.2f}, SWAP={adv_swap:.2f} -> SWAP")
            return Move(-1, -1)
            
        return self._get_fallback_move(board)

    def _choose_defensive_block(self, board: Board) -> Optional[Move]:
        """
        [大局观核心] 战略阻断。
        当对手 Dijkstra Cost 极低 (说明形成 VC) 时，寻找能最大化 Disrupt 的点。
        """
        opp_c = self.opp_colour()
        my_c = self.colour
        
        # 1. 基准 Cost
        base_opp_cost = self.connectivity_evaluator.shortest_path_cost(board, opp_c)
        
        # 只有对手很强时才触发 (Cost < 3.5)
        if base_opp_cost > 3.5:
            return None 

        # 2. 寻找最佳阻断点
        candidates = self._generate_local_candidates(board, opp_c, limit=30)
        best_block = None
        max_gain = 0.0
        
        for x, y in candidates:
            # [重要] 过滤无效切断 (不要去填双桥的眼)
            # 这里的 _is_inferior 会检查 Useless Cut
            if self.pattern_recognizer._is_inferior(board, x, y, my_c):
                continue
                
            # 模拟阻断
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, my_c)
            
            new_opp_cost = self.connectivity_evaluator.shortest_path_cost(b_sim, opp_c)
            gain = new_opp_cost - base_opp_cost
            
            # 如果增益显著 (例如破坏了 VC，Cost 至少增加 0.5)
            if gain > max_gain:
                max_gain = gain
                best_block = Move(x, y)
        
        if best_block and max_gain >= 0.5:
            return best_block
            
        return None

    def _postprocess_mcts_move(self, board: Board, move: Move) -> Move:
        """
        后处理：利用 PatternRecognizer 再次清洗 MCTS 结果。
        """
        if not move or (move.x == -1): return move
        
        # 检查是否选了 Inferior Move (如无效切断)
        if self.pattern_recognizer._is_inferior(board, move.x, move.y, self.colour):
            logger.warning(f"Postprocess: Override Inferior Move {move}")
            return self._find_best_alternative(board)
            
        return move

    def _find_best_alternative(self, board: Board) -> Move:
        """寻找替代走法 (基于 Light Score)"""
        best = self._get_first_valid_move(board)
        best_score = -999.0
        
        cands = self._generate_local_candidates(board, self.colour, limit=20)
        cands += self._generate_local_candidates(board, self.opp_colour(), limit=20)
        
        for x, y in set(cands):
            mv = Move(x, y)
            if not self._is_valid_move(mv, board): continue
            
            # 再次过滤 Inferior
            if self.pattern_recognizer._is_inferior(board, x, y, self.colour):
                continue
                
            s = self._score_move_light(board, mv)
            if s > best_score:
                best_score = s
                best = mv
        return best

    def _score_move_light(self, board: Board, move: Move) -> float:
        """
        轻量评分函数。
        结合 Dijkstra Cost 变化 和 Pattern Gamma 值。
        """
        if move.x == -1: return -99.0
        
        # 1. Inferior Check (快速剔除)
        if self.pattern_recognizer._is_inferior(board, move.x, move.y, self.colour):
            return -100.0

        # 2. Dijkstra Cost Gain
        my = self.colour
        opp = self.opp_colour()
        
        cur_my = self.connectivity_evaluator.shortest_path_cost(board, my)
        cur_opp = self.connectivity_evaluator.shortest_path_cost(board, opp)
        
        b2 = copy.deepcopy(board)
        b2.set_tile_colour(move.x, move.y, my)
        
        new_my = self.connectivity_evaluator.shortest_path_cost(b2, my)
        new_opp = self.connectivity_evaluator.shortest_path_cost(b2, opp)
        
        opp_gain = new_opp - cur_opp
        my_gain = cur_my - new_my
        
        # 3. Gamma Bonus (从 PatternRecognizer 获取战术评分)
        # get_prior 返回 [0, 1] 之间的值，可作为加分项
        gamma_score = self.pattern_recognizer.get_prior(board, move, my)
        
        # 综合评分: 阻断对手 > 自己连接 > 战术形状
        return opp_gain + 0.7 * my_gain + 0.5 * gamma_score

    # ==========================================
    # 基础工具与辅助
    # ==========================================

    def _update_modules_colour(self, colour: Colour):
        """Swap 后更新所有子模块颜色"""
        self.colour = colour
        self.pattern_recognizer.colour = colour
        self.threat_detector.colour = colour
        self.mcts_engine.colour = colour

    def _check_priority_moves(self, board: Board) -> Optional[Move]:
        """必杀/必防兜底"""
        # 检查必杀
        win_cands = self.threat_detector._get_candidates(board, self.colour)
        for x, y in win_cands:
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, self.colour)
            if b_sim.has_ended(self.colour): return Move(x, y)
            
        # 检查必防
        opp = self.opp_colour()
        lose_cands = self.threat_detector._get_candidates(board, opp)
        for x, y in lose_cands:
            # 过滤无效填眼
            if self.pattern_recognizer._is_inferior(board, x, y, self.colour):
                continue
            b_sim = copy.deepcopy(board)
            b_sim.set_tile_colour(x, y, opp)
            if b_sim.has_ended(opp): return Move(x, y)
            
        return None

    def _generate_local_candidates(self, board: Board, focus_colour: Colour, limit=40) -> List[Tuple[int, int]]:
        """生成局部候选点 (棋子周边 + 边缘关键点)"""
        candidates = set()
        size = board.size
        
        # 1. 棋子周边 (Radius 1)
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == focus_colour:
                    from src.Tile import Tile
                    for i in range(Tile.NEIGHBOUR_COUNT):
                        nx, ny = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
                        if 0 <= nx < size and 0 <= ny < size and board.tiles[nx][ny].colour is None:
                            candidates.add((nx, ny))
        
        # 2. 边缘关键点 (Edge Templates 区域)
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
        threats = self.threat_detector.detect_immediate_threats(board, self.opp_colour())
        if threats: return Move(threats[0][0], threats[0][1])
        return self._get_first_valid_move(board)

    def _get_first_valid_move(self, board: Board) -> Move:
        """兜底: 离中心最近的空位"""
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
        """安全兜底"""
        try:
            # 尝试 MCTS 缓存
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