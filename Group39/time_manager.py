"""
时间管理器 - MetaTimeManager
"""
import time
from typing import Optional
from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile

class TimeManager:
    """
    时间管理器
    
    功能：
    - 复杂度评估
    - 关键性评估
    - 时间分配
    - 时间跟踪
    """
    def __init__(self):
        self.start_time: Optional[float] = None
        self.total_time_limit = 270.0  # 4分半（秒）
        self.used_time = 0.0

    def _count_connections(self, board: Board, colour: Colour) -> int:
        """统计某一方的“直接连接边数”，用于衡量局面结构复杂度"""
        connections = 0
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == colour:
                    # 检查六个方向的邻居
                    for idx in range(Tile.NEIGHBOUR_COUNT):
                        x_n = x + Tile.I_DISPLACEMENTS[idx]
                        y_n = y + Tile.J_DISPLACEMENTS[idx]
                        if 0 <= x_n < size and 0 <= y_n < size:
                            if board.tiles[x_n][y_n].colour == colour:
                                connections += 1
        return connections // 2

    def _is_bridge_position(self, board: Board, x: int, y: int) -> bool:
        """判断 (x, y) 是否为“桥接位置”：周围同色或对方棋子较多，通常是关键点"""
        size = board.size
        red_count = 0
        blue_count = 0
        for idx in range(Tile.NEIGHBOUR_COUNT):
            x_n = x + Tile.I_DISPLACEMENTS[idx]
            y_n = y + Tile.J_DISPLACEMENTS[idx]
            if 0 <= x_n < size and 0 <= y_n < size:
                c = board.tiles[x_n][y_n].colour
                if c == Colour.RED:
                    red_count += 1
                elif c == Colour.BLUE:
                    blue_count += 1
        # 周围同一方或对方棋子数量>= 2就认为是关键桥位
        return red_count >= 2 or blue_count >= 2

    def _count_critical_positions(self, board: Board) -> int:
        """统计关键位置数量（桥接点、必争点）"""
        critical = 0
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    if self._is_bridge_position(board, x, y):
                        critical += 1
        return critical

    def _calculate_structural_complexity(self, board: Board) -> float:
        """计算局面“结构复杂度”（连接数 + 关键点），返回 0~1 之间的值"""
        red_conn = self._count_connections(board, Colour.RED)
        blue_conn = self._count_connections(board, Colour.BLUE)
        total_conn = red_conn + blue_conn
        critical_positions = self._count_critical_positions(board)
        score = 1.0
        # 连接数影响（more连接=more复杂）
        if total_conn > 15:
            score *= 1.3
        elif total_conn > 8:
            score *= 1.1
        # 关键位置影响
        if critical_positions > 5:
            score *= 1.2
        # 限制到[1.0,1.5]
        score = min(score, 1.5)
        # 归一化到[0,1]，1.0->0，1.5->1
        normalized = (score - 1.0) / 0.5
        return max(0.0, min(1.0, normalized))
    
    def start_timer(self):
        """开始计时"""
        self.start_time = time.time()
        self.used_time = 0.0
    
    def get_remaining_time(self) -> float:
        """
        获取剩余时间
        
        Returns:
            float: 剩余时间（秒）
        """
        if self.start_time is None:
            return self.total_time_limit
        
        elapsed = time.time() - self.start_time
        remaining = self.total_time_limit - elapsed
        return max(0.0, remaining)

    def _opposite_colour(self, c: Colour) -> Colour:
        return Colour.BLUE if c == Colour.RED else Colour.RED

    def assess_position_tension(self, board: Board, threat_detector=None, opp_colour=None) -> float:
        """
        评估“局面紧张度”（0~1）：
        - 越接近胜负越紧张（任意一方快连上了）
        - 双方连接成本差距越大越紧张（局面偏向一方）
        - 立即威胁（必杀/必防）越多越紧张

        只是判断该不该多花时间想，不用于选点
        """
        if threat_detector is None or opp_colour is None:
            return 0.0

        my_colour = self._opposite_colour(opp_colour)

        # 1) 立即威胁（双方都测一下，更稳）
        opp_threats = []
        my_threats = []
        try:
            opp_threats = threat_detector.detect_immediate_threats(board, opp_colour) or []
        except Exception:
            opp_threats = []
        try:
            my_threats = threat_detector.detect_immediate_threats(board, my_colour) or []
        except Exception:
            my_threats = []

        # threat_severity: 0~1
        # 有 WIN / LOSE 这种级别，直接拉满（因为你主代理里会规则层优先处理，但 TimeManager 需要知道“这回合很关键”）
        severe = 0
        for t in (opp_threats + my_threats):
            if len(t) >= 3 and (t[2] == "WIN" or t[2] == "LOSE"):
                severe += 1
        threat_severity = 1.0 if severe > 0 else min(1.0, (len(opp_threats) + len(my_threats)) / 4.0)

        # 2) 连接成本（接近连通越紧张；差距越大越紧张）
        closeness = 0.0
        gap = 0.0
        try:
            from .connectivity_evaluator import ConnectivityEvaluator
            ev = ConnectivityEvaluator()
            red_cost = ev.shortest_path_cost(board, Colour.RED)
            blue_cost = ev.shortest_path_cost(board, Colour.BLUE)

            # closeness：任意一方 cost 很低 -> 紧张
            finite_costs = [c for c in (red_cost, blue_cost) if c != float("inf")]
            if finite_costs:
                min_cost = min(finite_costs)
                # min_cost 越小越紧张；用 size 归一化
                closeness = max(0.0, min(1.0, 1.0 - (min_cost / max(1, board.size))))
            # gap：局面倾斜越大 -> 紧张（避免被碾/避免浪费时间）
            if red_cost != float("inf") and blue_cost != float("inf"):
                gap = max(0.0, min(1.0, abs(red_cost - blue_cost) / max(1, board.size)))
        except Exception:
            pass

        # 3) 综合（权重可解释且可调）
        tension = (
            0.45 * threat_severity +   # 有必杀必防时：最紧张
            0.35 * closeness +         # 接近终局时：更紧张
            0.20 * gap                 # 局面明显倾斜：也要更谨慎
        )
        return max(0.0, min(1.0, tension))
    
    def assess_position_complexity(self, board: Board, threat_detector=None, opp_colour=None) -> float:
        """
        评估局面复杂度
        
        优化：考虑更多因素，更精确的评估
        - 空位数（主要因素）
        - 威胁数（重要因素）
        - 游戏阶段（早期/中期/后期）
        - 连接性复杂度（双方连接成本差异）
        
        Args:
            board: 当前棋盘状态
            threat_detector: 威胁检测器
            opp_colour: 对手颜色
        
        Returns:
            float: 归一化的复杂度（0到1）
        """
        size = board.size
        total_tiles = size * size

        # 1. 统计空位数（主要因素）
        empty_tiles = sum(
            1
            for x in range(size)
            for y in range(size)
            if board.tiles[x][y].colour is None
        )
        empty_ratio = empty_tiles / total_tiles if total_tiles > 0 else 0.0
        
        # 2. 检测威胁数（重要因素）
        threat_count = 0
        if threat_detector and opp_colour:
            try:
                immediate_threats = threat_detector.detect_immediate_threats(board, opp_colour)
                threat_count = len(immediate_threats)
            except Exception:
                threat_count = 0
        
        # 3. 游戏阶段因子（早期更复杂，因为选择更多）
        # 早期（空位>70%）：复杂度高
        # 中期（30%<空位<70%）：复杂度中等
        # 后期（空位<30%）：复杂度低
        if empty_ratio > 0.7:
            phase_factor = 1.2  # 早期：复杂度略高
        elif empty_ratio < 0.3:
            phase_factor = 0.7  # 后期：相对简单
        else:
            phase_factor = 1.0  # 中期：正常
        
        # 4. 连接性复杂度（双方连接成本差异）
        connectivity_complexity = 0.0
        try:
            from .connectivity_evaluator import ConnectivityEvaluator
            evaluator = ConnectivityEvaluator()
            red_cost = evaluator.shortest_path_cost(board, Colour.RED)
            blue_cost = evaluator.shortest_path_cost(board, Colour.BLUE)

            if red_cost != float("inf") and blue_cost != float("inf"):
                avg_cost = (red_cost + blue_cost) / 2.0
                connectivity_complexity = max(0.0, min(1.0, 1.0 - avg_cost / max(1, size)))
        except Exception:
            connectivity_complexity = 0.0
        
        # 5. 结构复杂度
        structural_complexity = self._calculate_structural_complexity(board)
        # 空位因子：0~1，空越多，复杂度越高
        empty_factor = empty_ratio
        # 威胁因子：0~1，假设超过5个就视作 很多
        threat_factor = min(1.0, threat_count / 5.0)
        # phase_factor >1表示偏复杂，<1表示偏简单，这里用(phase_factor-0.7)/0.5粗略折算到0-1
        phase_component = (phase_factor - 0.7) / 0.5
        phase_component = max(0.0, min(1.0, phase_component))
        # 综合复杂度（权重总和约为1）
        complexity = (
                empty_factor * 0.30 +  # 空位30%
                threat_factor * 0.25 +  # 威胁25%
                connectivity_complexity * 0.20 +  # 连通性20%
                structural_complexity * 0.15 +  # 结构复杂度15%
                phase_component * 0.10  # 阶段因子10%
        )
        # 归一化到[0,1]
        normalized_complexity = max(0.0, min(1.0, complexity))
        return normalized_complexity
    
    def assess_position_criticality(self, board: Board, threat_detector=None, opp_colour=None) -> float:
        """
        评估局面关键性
        
        优化：更细粒度的关键性评估
        - WIN威胁：最高关键性（2.5）
        - LOSE威胁：高关键性（2.0）
        - 接近胜利/失败：中等关键性（1.5）
        - 正常情况：1.0
        
        Args:
            board: 当前棋盘状态
            threat_detector: 威胁检测器
            opp_colour: 对手颜色
        
        Returns:
            float: 关键性因子（1.0=正常，2.5=最高关键性）
        """

        immediate_threats = []  # 避免未定义

        # 1. 检查immediate threats（最高优先级）
        if threat_detector and opp_colour:
            try:
                immediate_threats = threat_detector.detect_immediate_threats(board, opp_colour) or []
            except Exception:
                immediate_threats = []

        if immediate_threats:
            # 检查是否有WIN威胁（立即获胜，最高优先级）
            win_threats = [t for t in immediate_threats if len(t) >= 3 and t[2] == "WIN"]
            if win_threats:
                return 2.5  # 最高关键性，需要更多时间确保正确
            # 检查是否有LOSE威胁（必须阻止）
            lose_threats = [t for t in immediate_threats if len(t) >= 3 and t[2] == "LOSE"]
            if lose_threats:
                return 2.0  # 高关键性，需要加倍时间
        
        # 2. 检查接近胜利/失败的情况
        # 使用连接性评估来判断是否接近胜利/失败
        try:
            from .connectivity_evaluator import ConnectivityEvaluator
            evaluator = ConnectivityEvaluator()
            
            # 计算双方的连接成本
            red_cost = evaluator.shortest_path_cost(board, Colour.RED)
            blue_cost = evaluator.shortest_path_cost(board, Colour.BLUE)

            # 用 opp_colour 区分：对手快连上更危险 -> 更关键
            if opp_colour is not None:
                my_colour = self._opposite_colour(opp_colour)

                opp_cost = red_cost if opp_colour == Colour.RED else blue_cost
                my_cost = red_cost if my_colour == Colour.RED else blue_cost

                if opp_cost != float("inf") and opp_cost <= 2:
                    return 1.6  # 对手快赢：更关键
                if my_cost != float("inf") and my_cost <= 2:
                    return 1.4  # 我快赢：也关键，但稍低一点
        except Exception:
            pass
        
        # 3. 后期提高关键性
        empty_tiles = sum(
            1 for x in range(board.size)
            for y in range(board.size)
            if board.tiles[x][y].colour is None
        )
        total_tiles = board.size * board.size
        empty_ratio = empty_tiles / total_tiles if total_tiles > 0 else 0.5
        # 后期（空位<30%）：更关键，需要更仔细
        if empty_ratio < 0.3:
            return 1.2  # 后期：稍微提高关键性
        
        # 正常情况
        return 1.0
    
    def allocate_time(self, board: Board, total_time_remaining: float, 
                      total_turns_remaining: int, threat_detector=None, 
                      opp_colour=None) -> float:
        """
        base_time（均分） * stage_factor（中后盘更愿意花时间）
                            * (1 + a*complexity + b*tension) * criticality
        然后再做安全缓冲 + 上下限裁剪。
        """
        # 基础时间 = 剩余时间 / 剩余回合数
        if total_turns_remaining <= 0:
            total_turns_remaining = 1  # 防止除零

        A_COMPLEXITY = 0.55  # 复杂度影响
        B_TENSION = 0.85  # 紧张度影响（比复杂度更“值”）
        # stage_factor：早期<1，中期≈1，后期>1
        STAGE_EARLY = 0.85
        STAGE_MID = 1.00
        STAGE_LATE = 1.20

        # 安全缓冲（越缺时间越保守）
        BUFFER_RICH = 0.95  # >60s
        BUFFER_MID = 0.90  # 30~60s
        BUFFER_LOW = 0.82  # <30s

        MAX_SINGLE_SEARCH_TIME = 10.0  # 单次不超过10s（防止一把梭哈）

        base_time = total_time_remaining / total_turns_remaining

        # 游戏阶段（用空位比例判断）
        size = board.size
        total_tiles = size * size
        empty_tiles = sum(
            1 for x in range(size)
            for y in range(size)
            if board.tiles[x][y].colour is None
        )
        empty_ratio = empty_tiles / total_tiles if total_tiles > 0 else 0.5

        if empty_ratio > 0.70:
            stage_factor = STAGE_EARLY
            max_time_ratio = 0.25
        elif empty_ratio < 0.30:
            stage_factor = STAGE_LATE
            max_time_ratio = 0.50
        else:
            stage_factor = STAGE_MID
            max_time_ratio = 0.35

        complexity = self.assess_position_complexity(board, threat_detector, opp_colour)
        criticality = self.assess_position_criticality(board, threat_detector, opp_colour)
        tension = self.assess_position_tension(board, threat_detector, opp_colour)

        # 主公式：解释性强、参数可调
        multiplier = stage_factor * (1.0 + A_COMPLEXITY * complexity + B_TENSION * tension)
        time_budget = base_time * multiplier * criticality

        # 安全缓冲
        if total_time_remaining > 60.0:
            time_budget *= BUFFER_RICH
        elif total_time_remaining > 30.0:
            time_budget *= BUFFER_MID
        else:
            time_budget *= BUFFER_LOW

        # 上限：不能超过剩余时间的一定比例
        time_budget = min(time_budget, total_time_remaining * max_time_ratio)

        # 下限：复杂/紧张局面至少给一点像样的时间
        min_time = 0.012 + 0.040 * tension + 0.020 * complexity  # 大概 0.012~0.072
        time_budget = max(time_budget, min_time)

        # 绝对上限
        time_budget = min(time_budget, MAX_SINGLE_SEARCH_TIME)
        
        return time_budget
    
    def _detect_potential_kills(self, board: Board) -> list:
        """
        检测2步内可赢的情况
        
        Args:
            board: 当前棋盘状态
        
        Returns:
            list: 潜在胜利点列表
        """
        kills = []
        return kills
