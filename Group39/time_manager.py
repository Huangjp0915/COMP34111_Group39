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
        self.total_time_limit = 600
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
        if threat_detector:
            try:
                from .connectivity_evaluator import ConnectivityEvaluator
                evaluator = ConnectivityEvaluator()
                red_cost = evaluator.shortest_path_cost(board, Colour.RED)
                blue_cost = evaluator.shortest_path_cost(board, Colour.BLUE)
                if red_cost != float("inf") and blue_cost != float("inf"):
                    avg_cost = (red_cost + blue_cost) / 2
                    connectivity_complexity = max(0.0, 1.0 - avg_cost / size) * 0.3
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
        # 1. 检查immediate threats（最高优先级）
        if threat_detector and opp_colour:
            immediate_threats = threat_detector.detect_immediate_threats(
                board, opp_colour
            )
            if immediate_threats:
                # 检查是否有WIN威胁（立即获胜，最高优先级）
                win_threats = [t for t in immediate_threats if t[2] == "WIN"]
                if win_threats:
                    return 2.5  # 最高关键性，需要更多时间确保正确
                
                # 检查是否有LOSE威胁（必须阻止）
                lose_threats = [t for t in immediate_threats if t[2] == "LOSE"]
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
            
            # 如果一方非常接近连接（cost <= 2），关键性提高
            if red_cost != float('inf') and red_cost <= 2:
                return 1.5  # 接近胜利，中等关键性
            if blue_cost != float('inf') and blue_cost <= 2:
                return 1.5  # 接近失败，中等关键性
        except:
            pass  # 如果无法计算，忽略
        
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
        智能分配时间
        
        优化：更精确的时间分配，减少时间浪费
        - 动态调整复杂度因子
        - 根据游戏阶段调整时间分配
        - 更智能的安全缓冲
        
        Args:
            board: 当前棋盘状态
            total_time_remaining: 剩余总时间
            total_turns_remaining: 剩余回合数
            threat_detector: 威胁检测器（可选）
            opp_colour: 对手颜色（可选）
        
        Returns:
            float: 分配的时间预算（秒）
        """
        # 基础时间 = 剩余时间 / 剩余回合数
        if total_turns_remaining <= 0:
            total_turns_remaining = 1  # 防止除零
        
        base_time = total_time_remaining / total_turns_remaining
        
        # 评估复杂度和关键性
        complexity = self.assess_position_complexity(board, threat_detector, opp_colour)
        criticality = self.assess_position_criticality(
            board, threat_detector, opp_colour
        )
        
        # 优化：动态复杂度因子（根据游戏阶段调整）
        empty_tiles = sum(
            1 for x in range(board.size)
            for y in range(board.size)
            if board.tiles[x][y].colour is None
        )
        total_tiles = board.size * board.size
        empty_ratio = empty_tiles / total_tiles if total_tiles > 0 else 0.5
        
        # 早期：复杂度影响较小（选择多，但不需要太深入）
        # 后期：复杂度影响较大（选择少，但需要更仔细）
        if empty_ratio > 0.7:
            # 早期：复杂度因子较小（0.3倍）
            complexity_factor = 1.0 + complexity * 0.3
        elif empty_ratio < 0.3:
            # 后期：复杂度因子较大（0.8倍）
            complexity_factor = 1.0 + complexity * 0.8
        else:
            # 中期：正常复杂度因子（0.5倍）
            complexity_factor = 1.0 + complexity * 0.5
        
        # 关键性因子（遇到威胁时增加时间）
        time_budget = base_time * complexity_factor * criticality
        
        # 优化：动态安全缓冲（根据剩余时间调整）
        # 如果剩余时间充足，可以预留更多缓冲
        # 如果剩余时间紧张，减少缓冲
        if total_time_remaining > 60.0:
            # 时间充足：预留15%缓冲
            safety_buffer = 0.85
        elif total_time_remaining > 30.0:
            # 时间中等：预留10%缓冲
            safety_buffer = 0.9
        else:
            # 时间紧张：预留5%缓冲
            safety_buffer = 0.95
        
        time_budget = time_budget * safety_buffer
        
        # 优化：更智能的时间上限
        # 根据剩余回合数调整上限
        if total_turns_remaining > 20:
            # 还有很多回合：最多使用剩余时间的60%
            max_time_ratio = 0.6
        elif total_turns_remaining > 10:
            # 中等回合数：最多使用剩余时间的70%
            max_time_ratio = 0.7
        else:
            # 接近结束：最多使用剩余时间的80%
            max_time_ratio = 0.8
        
        time_budget = min(time_budget, total_time_remaining * max_time_ratio)
        
        # 优化：最小时间保证（根据复杂度调整）
        # 复杂局面需要更多时间
        min_time = 0.01 + complexity * 0.05  # 最小0.01秒，最多0.06秒
        time_budget = max(time_budget, min_time)
        
        # 优化：最大时间限制（避免单次搜索时间过长）
        # 即使很复杂，单次搜索也不应该超过10秒
        max_single_search_time = 10.0
        time_budget = min(time_budget, max_single_search_time)
        
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

