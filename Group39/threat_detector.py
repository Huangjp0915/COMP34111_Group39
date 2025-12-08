"""
威胁检测模块
"""
import copy
from typing import List, Tuple

from src.Board import Board
from src.Colour import Colour


class ThreatDetector:
    """
    威胁检测器
    
    功能：
    - 检测直接威胁（immediate threats）
    - 检测简单模式（simple patterns）
    """
    
    def __init__(self, colour: Colour):
        self.colour = colour
    
    def detect_immediate_threats(self, board: Board, opponent_colour: Colour) -> List[Tuple[int, int, str]]:
        """
        检测一手输/赢的情况
        
        优化：减少不必要的模拟，使用快速预检查
        
        优先级：WIN > LOSE（先检查我方能否赢，再检查对手能否赢）
        
        Args:
            board: 当前棋盘状态
            opponent_colour: 对手颜色
        
        Returns:
            List[Tuple[int, int, str]]: [(x, y, threat_type), ...]
            threat_type: "WIN" 或 "LOSE"
        """
        threats = []
        
        # 优化：快速预检查 - 只检查可能形成连接的位置
        # 1. 优先检查我方是否能在下一步赢（WIN）
        win_candidates = self._get_win_candidates(board, self.colour)
        for x, y in win_candidates:
            if board.tiles[x][y].colour is None:
                # 优化：使用快速检查，避免不必要的深拷贝
                if self._quick_win_check(board, x, y, self.colour):
                    threats.append((x, y, "WIN"))
                    # 如果找到胜利点，可以立即返回（最高优先级）
                    return threats
        
        # 2. 检查对手是否能在下一步赢（我方输）
        lose_candidates = self._get_win_candidates(board, opponent_colour)
        for x, y in lose_candidates:
            if board.tiles[x][y].colour is None:
                # 优化：使用快速检查
                if self._quick_win_check(board, x, y, opponent_colour):
                    threats.append((x, y, "LOSE"))
        
        return threats
    
    def _get_win_candidates(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        获取可能形成连接的位置（快速预检查）
        
        优化：只检查接近边界或接近已有棋子的位置
        
        Args:
            board: 当前棋盘状态
            colour: 要检查的颜色
        
        Returns:
            List[Tuple[int, int]]: 候选位置列表
        """
        candidates = []
        
        if colour == Colour.RED:
            # RED: 检查接近顶部和底部的行
            for y in range(board.size):
                # 顶部行附近（0-2行）
                for x in range(min(3, board.size)):
                    if board.tiles[x][y].colour is None:
                        candidates.append((x, y))
                # 底部行附近（size-3到size-1行）
                for x in range(max(0, board.size - 3), board.size):
                    if board.tiles[x][y].colour is None:
                        candidates.append((x, y))
        else:  # BLUE
            # BLUE: 检查接近左边和右边的列
            for x in range(board.size):
                # 左列附近（0-2列）
                for y in range(min(3, board.size)):
                    if board.tiles[x][y].colour is None:
                        candidates.append((x, y))
                # 右列附近（size-3到size-1列）
                for y in range(max(0, board.size - 3), board.size):
                    if board.tiles[x][y].colour is None:
                        candidates.append((x, y))
        
        # 去重
        return list(set(candidates))
    
    def _quick_win_check(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        快速检查在(x,y)下子是否能形成连接
        
        优化：避免深拷贝，直接检查
        
        Args:
            board: 当前棋盘状态
            x, y: 位置坐标
            colour: 颜色
        
        Returns:
            bool: 是否能形成连接
        """
        # 临时设置棋子
        original_colour = board.tiles[x][y].colour
        board.tiles[x][y].colour = colour
        
        # 重置winner状态
        board._winner = None
        
        # 检查是否形成连接
        result = board.has_ended(colour)
        
        # 恢复原状态
        board.tiles[x][y].colour = original_colour
        board._winner = None
        
        return result
    
    def detect_simple_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        """
        检测少量固定桥接点/关键连接点
        
        增强：添加更多模式识别
        
        Args:
            board: 当前棋盘状态
            colour: 要检测的颜色
        
        Returns:
            List[Tuple[int, int, float]]: [(x, y, weight), ...]
        """
        patterns = []
        
        # 1. 桥接点检测（权重0.8）
        bridge_points = self._detect_bridge_patterns(board, colour)
        patterns.extend([(x, y, 0.8) for x, y in bridge_points])
        
        # 2. 连接点检测（权重0.6）
        connection_points = self._detect_connection_patterns(board, colour)
        patterns.extend([(x, y, 0.6) for x, y in connection_points])
        
        # 3. 新增：阻塞点检测（权重0.7）- 阻止对手形成连接的关键点
        blocking_points = self._detect_blocking_patterns(board, colour)
        patterns.extend([(x, y, 0.7) for x, y in blocking_points])
        
        # 4. 新增：扩展点检测（权重0.5）- 扩展已有连接的点
        extension_points = self._detect_extension_patterns(board, colour)
        patterns.extend([(x, y, 0.5) for x, y in extension_points])
        
        # 5. 新增：双威胁点检测（权重0.9）- 同时形成两个威胁的点
        double_threat_points = self._detect_double_threat_patterns(board, colour)
        patterns.extend([(x, y, 0.9) for x, y in double_threat_points])
        
        return patterns
    
    def _detect_bridge_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        检测桥接点：连接两个分离棋子组的关键点
        
        优化：减少不必要的深拷贝，使用快速检查
        
        Args:
            board: 当前棋盘状态
            colour: 要检测的颜色
        
        Returns:
            List[Tuple[int, int]]: 桥接点列表
        """
        bridge_points = []
        
        # 找到所有我方棋子位置
        my_pieces = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour == colour:
                    my_pieces.append((x, y))
        
        # 如果棋子少于2个，不需要桥接
        if len(my_pieces) < 2:
            return bridge_points
        
        # 优化：只检查接近我方棋子的空位（2步内）
        candidate_positions = set()
        for px, py in my_pieces:
            from src.Tile import Tile
            for i in range(Tile.NEIGHBOUR_COUNT):
                dx = Tile.I_DISPLACEMENTS[i]
                dy = Tile.J_DISPLACEMENTS[i]
                for dist in range(1, 3):  # 1-2步内
                    nx, ny = px + dx * dist, py + dy * dist
                    if 0 <= nx < board.size and 0 <= ny < board.size:
                        if board.tiles[nx][ny].colour is None:
                            candidate_positions.add((nx, ny))
        
        # 检查候选位置是否是桥接点
        for x, y in candidate_positions:
            # 优化：使用快速检查，避免深拷贝
            original_colour = board.tiles[x][y].colour
            board.tiles[x][y].colour = colour
            
            # 检查是否能连接多个分离的组
            neighbor_groups = self._count_neighbor_groups(board, x, y, colour)
            if neighbor_groups >= 2:
                bridge_points.append((x, y))
            
            # 恢复原状态
            board.tiles[x][y].colour = original_colour
        
        return bridge_points
    
    def _count_neighbor_groups(self, board: Board, x: int, y: int, colour: Colour) -> int:
        """
        计算位置(x,y)的邻居中有多少个不同的棋子组
        
        Args:
            board: 棋盘状态
            x, y: 位置坐标
            colour: 颜色
        
        Returns:
            int: 邻居中的组数
        """
        from src.Tile import Tile
        
        # 使用简单的连通性检查
        visited = set()
        groups = 0
        
        for i in range(Tile.NEIGHBOUR_COUNT):
            dx = Tile.I_DISPLACEMENTS[i]
            dy = Tile.J_DISPLACEMENTS[i]
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < board.size and 0 <= ny < board.size:
                if board.tiles[nx][ny].colour == colour and (nx, ny) not in visited:
                    # 找到一个新的组，标记所有连通的棋子
                    self._mark_connected_group(board, nx, ny, colour, visited)
                    groups += 1
        
        return groups
    
    def _mark_connected_group(self, board: Board, start_x: int, start_y: int, 
                              colour: Colour, visited: set):
        """
        标记从(start_x, start_y)开始的所有连通棋子
        
        Args:
            board: 棋盘状态
            start_x, start_y: 起始位置
            colour: 颜色
            visited: 已访问集合
        """
        from collections import deque
        from src.Tile import Tile
        
        queue = deque([(start_x, start_y)])
        visited.add((start_x, start_y))
        
        while queue:
            x, y = queue.popleft()
            
            for i in range(Tile.NEIGHBOUR_COUNT):
                dx = Tile.I_DISPLACEMENTS[i]
                dy = Tile.J_DISPLACEMENTS[i]
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < board.size and 0 <= ny < board.size and
                    board.tiles[nx][ny].colour == colour and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    def _detect_connection_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        检测接近目标边界的连接点
        
        优化：减少搜索范围，只检查关键区域
        
        Args:
            board: 当前棋盘状态
            colour: 要检测的颜色
        
        Returns:
            List[Tuple[int, int]]: 连接点列表
        """
        connection_points = []
        
        if colour == Colour.RED:
            # RED: 接近顶部和底部边界
            # 优化：只检查边界附近的关键区域
            boundary_rows = [0, 1, 2, board.size - 3, board.size - 2, board.size - 1]
            for x in boundary_rows:
                if 0 <= x < board.size:
                    for y in range(board.size):
                        if board.tiles[x][y].colour is None:
                            # 检查是否接近我方棋子（2步内）
                            if self._is_near_my_pieces(board, x, y, colour, distance=2):
                                connection_points.append((x, y))
        else:  # BLUE
            # BLUE: 接近左边和右边边界
            boundary_cols = [0, 1, 2, board.size - 3, board.size - 2, board.size - 1]
            for y in boundary_cols:
                if 0 <= y < board.size:
                    for x in range(board.size):
                        if board.tiles[x][y].colour is None:
                            if self._is_near_my_pieces(board, x, y, colour, distance=2):
                                connection_points.append((x, y))
        
        return connection_points
    
    def _is_near_my_pieces(self, board: Board, x: int, y: int, colour: Colour, distance: int = 2) -> bool:
        """
        检查位置是否接近我方棋子（在distance步内）
        
        使用BFS检查distance步内是否有我方棋子
        
        Args:
            board: 当前棋盘状态
            x, y: 位置坐标
            colour: 我方颜色
            distance: 距离阈值
        
        Returns:
            bool: 是否接近
        """
        from collections import deque
        from src.Tile import Tile
        
        if distance <= 0:
            return False
        
        queue = deque([(x, y, 0)])
        visited = {(x, y)}
        
        while queue:
            cx, cy, dist = queue.popleft()
            
            # 如果距离超过阈值，停止搜索
            if dist >= distance:
                continue
            
            # 检查6个邻居
            for i in range(Tile.NEIGHBOUR_COUNT):
                dx = Tile.I_DISPLACEMENTS[i]
                dy = Tile.J_DISPLACEMENTS[i]
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < board.size and 0 <= ny < board.size:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        
                        # 如果找到我方棋子，返回True
                        if board.tiles[nx][ny].colour == colour:
                            return True
                        
                        # 如果是空位，继续搜索
                        if board.tiles[nx][ny].colour is None:
                            queue.append((nx, ny, dist + 1))
        
        return False
    
    def _detect_blocking_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        检测阻塞点：阻止对手形成连接的关键点
        
        Args:
            board: 当前棋盘状态
            colour: 我方颜色
        
        Returns:
            List[Tuple[int, int]]: 阻塞点列表
        """
        blocking_points = []
        opponent_colour = Colour.opposite(colour)
        
        # 优化：只检查对手可能形成连接的关键位置
        # 检查对手接近边界的位置
        opponent_candidates = self._get_win_candidates(board, opponent_colour)
        
        for x, y in opponent_candidates:
            if board.tiles[x][y].colour is None:
                # 检查如果对手在这里下子，是否接近连接
                if self._is_critical_for_opponent(board, x, y, opponent_colour):
                    blocking_points.append((x, y))
        
        return blocking_points
    
    def _is_critical_for_opponent(self, board: Board, x: int, y: int, opponent_colour: Colour) -> bool:
        """
        检查位置是否对对手形成连接很关键
        
        Args:
            board: 当前棋盘状态
            x, y: 位置坐标
            opponent_colour: 对手颜色
        
        Returns:
            bool: 是否关键
        """
        # 优化：使用快速检查，避免深拷贝
        original_colour = board.tiles[x][y].colour
        board.tiles[x][y].colour = opponent_colour
        
        # 检查对手的连接性
        from .connectivity_evaluator import ConnectivityEvaluator
        evaluator = ConnectivityEvaluator()
        
        # 计算对手下子后的连接成本
        after_cost = evaluator.shortest_path_cost(board, opponent_colour)
        
        # 恢复原状态
        board.tiles[x][y].colour = original_colour
        
        # 如果对手下子后距离连接很近（cost <= 2），这个位置很关键
        return after_cost <= 2
    
    def _detect_extension_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        检测扩展点：扩展已有连接的点
        
        Args:
            board: 当前棋盘状态
            colour: 要检测的颜色
        
        Returns:
            List[Tuple[int, int]]: 扩展点列表
        """
        extension_points = []
        
        # 找到所有我方棋子
        my_pieces = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour == colour:
                    my_pieces.append((x, y))
        
        if len(my_pieces) == 0:
            return extension_points
        
        # 优化：只检查接近我方棋子的空位（2步内）
        candidate_positions = set()
        for px, py in my_pieces:
            from src.Tile import Tile
            for i in range(Tile.NEIGHBOUR_COUNT):
                dx = Tile.I_DISPLACEMENTS[i]
                dy = Tile.J_DISPLACEMENTS[i]
                for dist in range(1, 3):  # 1-2步内
                    nx, ny = px + dx * dist, py + dy * dist
                    if 0 <= nx < board.size and 0 <= ny < board.size:
                        if board.tiles[nx][ny].colour is None:
                            candidate_positions.add((nx, ny))
        
        # 检查候选位置是否能改善连接性
        for x, y in candidate_positions:
            if self._improves_connectivity(board, x, y, colour):
                extension_points.append((x, y))
        
        return extension_points
    
    def _improves_connectivity(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        检查在(x,y)下子是否能改善连接性
        
        Args:
            board: 当前棋盘状态
            x, y: 位置坐标
            colour: 颜色
        
        Returns:
            bool: 是否能改善
        """
        from .connectivity_evaluator import ConnectivityEvaluator
        evaluator = ConnectivityEvaluator()
        
        # 计算下子前的连接成本
        before_cost = evaluator.shortest_path_cost(board, colour)
        
        # 优化：使用快速检查，避免深拷贝
        original_colour = board.tiles[x][y].colour
        board.tiles[x][y].colour = colour
        
        # 计算下子后的连接成本
        after_cost = evaluator.shortest_path_cost(board, colour)
        
        # 恢复原状态
        board.tiles[x][y].colour = original_colour
        
        # 如果成本降低，说明改善了连接性
        return after_cost < before_cost
    
    def _detect_double_threat_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        检测双威胁点：同时形成两个威胁的点
        
        Args:
            board: 当前棋盘状态
            colour: 要检测的颜色
        
        Returns:
            List[Tuple[int, int]]: 双威胁点列表
        """
        double_threat_points = []
        
        # 优化：只检查接近我方棋子的空位
        my_pieces = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour == colour:
                    my_pieces.append((x, y))
        
        if len(my_pieces) == 0:
            return double_threat_points
        
        # 获取候选位置（接近我方棋子）
        candidate_positions = set()
        for px, py in my_pieces:
            from src.Tile import Tile
            for i in range(Tile.NEIGHBOUR_COUNT):
                dx = Tile.I_DISPLACEMENTS[i]
                dy = Tile.J_DISPLACEMENTS[i]
                for dist in range(1, 3):  # 1-2步内
                    nx, ny = px + dx * dist, py + dy * dist
                    if 0 <= nx < board.size and 0 <= ny < board.size:
                        if board.tiles[nx][ny].colour is None:
                            candidate_positions.add((nx, ny))
        
        # 检查每个候选位置
        for x, y in candidate_positions:
            # 计算在这个位置下子后形成的威胁数
            threat_count = self._count_threats_after_move(board, x, y, colour)
            if threat_count >= 2:
                double_threat_points.append((x, y))
        
        return double_threat_points
    
    def _count_threats_after_move(self, board: Board, x: int, y: int, colour: Colour) -> int:
        """
        计算在(x,y)下子后形成的威胁数
        
        Args:
            board: 当前棋盘状态
            x, y: 位置坐标
            colour: 颜色
        
        Returns:
            int: 威胁数
        """
        # 优化：使用快速检查，避免深拷贝
        original_colour = board.tiles[x][y].colour
        board.tiles[x][y].colour = colour
        
        threat_count = 0
        
        # 检查是否形成桥接点
        neighbor_groups = self._count_neighbor_groups(board, x, y, colour)
        if neighbor_groups >= 2:
            threat_count += 1
        
        # 检查是否接近目标边界
        if self._is_near_target_boundary(board, x, y, colour):
            threat_count += 1
        
        # 恢复原状态
        board.tiles[x][y].colour = original_colour
        
        return threat_count
    
    def _is_near_target_boundary(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        检查位置是否接近目标边界
        
        Args:
            board: 当前棋盘状态
            x, y: 位置坐标
            colour: 颜色
        
        Returns:
            bool: 是否接近
        """
        if colour == Colour.RED:
            # RED: 接近顶部或底部
            return x <= 2 or x >= board.size - 3
        else:  # BLUE
            # BLUE: 接近左边或右边
            return y <= 2 or y >= board.size - 3

