"""
模式识别器
"""
import copy
from collections import deque
from typing import List, Tuple

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile


class PatternRecognizer:
    """
    模式识别器（rule-based版本）
    
    功能：
    - 检测桥接点
    - 检测连接点
    - 为MCTS节点初始化prior概率
    """
    
    def __init__(self, colour: Colour):
        self.colour = colour
    
    def detect_bridge_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        检测桥接点：连接两个分离棋子组的关键点
        
        简化实现：检测空位，如果放置该位置的棋子后能连接两个或多个分离的棋子组
        
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
        
        # 检查每个空位是否是桥接点
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    # 模拟放置棋子
                    test_board = copy.deepcopy(board)
                    test_board.set_tile_colour(x, y, colour)
                    
                    # 检查是否能连接多个分离的组
                    # 简化：检查该位置的邻居中是否有多个我方棋子组
                    neighbor_groups = self._count_neighbor_groups(test_board, x, y, colour)
                    if neighbor_groups >= 2:
                        bridge_points.append((x, y))
        
        return bridge_points
    
    def detect_connection_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        检测接近目标边界的连接点
        
        Args:
            board: 当前棋盘状态
            colour: 要检测的颜色
        
        Returns:
            List[Tuple[int, int]]: 连接点列表
        """
        connection_points = []
        
        if colour == Colour.RED:
            # RED: 接近底部边界（row = size-1）
            target_row = board.size - 1
            for y in range(board.size):
                # 检查目标行附近3行
                for x in range(max(0, target_row - 2), min(board.size, target_row + 1)):
                    if board.tiles[x][y].colour is None:
                        # 检查是否接近我方棋子（2步内）
                        if self._is_near_my_pieces(board, x, y, colour, distance=2):
                            connection_points.append((x, y))
        else:  # BLUE
            # BLUE: 接近右边界（col = size-1）
            target_col = board.size - 1
            for x in range(board.size):
                for y in range(max(0, target_col - 2), min(board.size, target_col + 1)):
                    if board.tiles[x][y].colour is None:
                        if self._is_near_my_pieces(board, x, y, colour, distance=2):
                            connection_points.append((x, y))
        
        return connection_points
    
    def detect_simple_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        """
        检测简单模式（桥接点+连接点）
        
        Args:
            board: 当前棋盘状态
            colour: 要检测的颜色
        
        Returns:
            List[Tuple[int, int, float]]: [(x, y, weight), ...]
        """
        patterns = []
        
        # 桥接点
        bridge_points = self.detect_bridge_patterns(board, colour)
        patterns.extend([(x, y, 0.8) for x, y in bridge_points])
        
        # 连接点
        connection_points = self.detect_connection_patterns(board, colour)
        patterns.extend([(x, y, 0.6) for x, y in connection_points])
        
        return patterns
    
    def get_prior(self, board: Board, move: Move, colour: Colour) -> float:
        """
        为MCTS子节点初始化prior概率
        基于模式识别
        
        Args:
            board: 当前棋盘状态
            move: 走法
            colour: 当前玩家颜色
        
        Returns:
            float: prior概率（0到1）
        """
        prior = 0.1  # 基础prior
        
        # 检查是否是桥接点
        bridge_points = self.detect_bridge_patterns(board, colour)
        if (move.x, move.y) in bridge_points:
            prior += 0.3
        
        # 检查是否是连接点
        connection_points = self.detect_connection_patterns(board, colour)
        if (move.x, move.y) in connection_points:
            prior += 0.2
        
        # 归一化到[0, 1]
        return min(1.0, prior)
    
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

