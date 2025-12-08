"""
连接性评估模块
"""
from collections import deque
from typing import Optional

from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile


class ConnectivityEvaluator:
    """
    连接性评估器
    
    功能：
    - 计算最短路径成本
    - 评估叶子节点
    """
    
    def shortest_path_cost(self, board: Board, colour: Colour) -> float:
        """
        计算从一边到另一边的最短路径成本
        
        - 自己棋子：cost = 0（已连接）
        - 空位：cost = 1（需要占据）
        - 对手棋子：不可通过
        
        Args:
            board: 当前棋盘状态
            colour: 要评估的颜色
        
        Returns:
            float: 最短路径成本（如果无法连接返回inf）
        """
        # 快速检查：如果已经连接，直接返回0
        if board.has_ended(colour):
            return 0.0
        
        if colour == Colour.RED:
            # RED: 从顶部（row=0）到底部（row=size-1）
            start_nodes = [(0, y) for y in range(board.size)]
            end_condition = lambda x, y: x == board.size - 1
        else:  # BLUE
            # BLUE: 从左（col=0）到右（col=size-1）
            start_nodes = [(x, 0) for x in range(board.size)]
            end_condition = lambda x, y: y == board.size - 1
        
        # BFS搜索（使用优先队列）
        # 但由于cost只有0和1，可以使用两个队列或直接BFS
        queue = deque()
        visited = set()
        
        # 初始化：所有起始节点
        for start in start_nodes:
            tile = board.tiles[start[0]][start[1]]
            if tile.colour == colour:
                # 起始节点是我方棋子，cost=0
                queue.append((start, 0))
                visited.add(start)
            elif tile.colour is None:
                # 起始节点是空位，cost=1
                queue.append((start, 1))
                visited.add(start)
            # 对手棋子：跳过
        
        # BFS主循环
        while queue:
            (x, y), cost = queue.popleft()
            
            # 检查是否到达目标边界
            if end_condition(x, y):
                return float(cost)
            
            # 检查6个邻居（Hex的邻居）
            for i in range(Tile.NEIGHBOUR_COUNT):
                dx = Tile.I_DISPLACEMENTS[i]
                dy = Tile.J_DISPLACEMENTS[i]
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < board.size and 0 <= ny < board.size:
                    if (nx, ny) not in visited:
                        neighbor_tile = board.tiles[nx][ny]
                        
                        if neighbor_tile.colour == colour:
                            # 自己的棋子，cost不变（优先处理cost=0的节点）
                            new_cost = cost
                            # 将cost=0的节点放在队列前面
                            queue.appendleft(((nx, ny), new_cost))
                        elif neighbor_tile.colour is None:
                            # 空位，cost+1
                            new_cost = cost + 1
                            queue.append(((nx, ny), new_cost))
                        else:
                            # 对手棋子，跳过
                            continue
                        
                        visited.add((nx, ny))
        
        # 无法连接
        return float('inf')
    
    def evaluate_leaf(self, board: Board, colour: Colour) -> float:
        """
        MCTS模拟到叶子节点时的评估
        使用连接性评估作为启发式
        
        Args:
            board: 当前棋盘状态
            colour: 我方颜色
        
        Returns:
            float: 评估值（-1到1）
        """
        my_cost = self.shortest_path_cost(board, colour)
        opp_cost = self.shortest_path_cost(board, Colour.opposite(colour))
        
        # 处理inf情况
        if my_cost == float('inf') and opp_cost == float('inf'):
            return 0.0  # 双方都无法连接，平局
        elif my_cost == float('inf'):
            return -1.0  # 我方无法连接，对手可以
        elif opp_cost == float('inf'):
            return 1.0  # 对手无法连接，我方可以
        
        # 评估：对手成本 - 我方成本
        # 值越大，我方越有利
        score = opp_cost - my_cost
        
        # 归一化到[-1, 1]
        # 假设最大成本为board.size（最坏情况：全是空位）
        max_cost = board.size
        if max_cost > 0:
            normalized_score = score / (max_cost * 2)
        else:
            normalized_score = 0.0
        
        # 限制在[-1, 1]
        return max(-1.0, min(1.0, normalized_score))

