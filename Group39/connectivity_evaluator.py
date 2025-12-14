"""
connectivity_evaluator.py
[Strategy Enhanced Version v8.0]
基于 Dijkstra + 虚拟连接 (VC) + 动态势能场 (Potential Field) 的评估器。

修改重点：
1. evaluate_leaf: 调整了评分公式权重。
   Score = (Opp_Cost * 0.8) - (My_Cost * 1.2)
   这意味着 MCTS 会认为 "缩短自己的路径" 比 "单纯增加对手路径" 更有价值。
   从而解决 "只顾阻断对手，自己却死掉" 的问题。
"""

from __future__ import annotations
import heapq
from typing import List, Tuple, Set

from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile

# 无穷大常量
INF = 10**9

class ConnectivityEvaluator:
    """
    高级连接性评估器
    负责计算考虑了战术局势的"广义距离"。
    """

    def __init__(self):
        # 预计算桥接模板
        # 格式: (target_dx, target_dy, [(mid1_dx, mid1_dy), (mid2_dx, mid2_dy)])
        self.bridge_templates = self._precompute_bridge_templates()

    def _precompute_bridge_templates(self) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
        """
        预计算 Hex 网格上的马步 (Bridge) 偏移量。
        """
        templates = []
        for i in range(Tile.NEIGHBOUR_COUNT):
            v1_x, v1_y = Tile.I_DISPLACEMENTS[i], Tile.J_DISPLACEMENTS[i]
            next_i = (i + 1) % Tile.NEIGHBOUR_COUNT
            v2_x, v2_y = Tile.I_DISPLACEMENTS[next_i], Tile.J_DISPLACEMENTS[next_i]
            target_dx = v1_x + v2_x
            target_dy = v1_y + v2_y
            mid_points = [(v1_x, v1_y), (v2_x, v2_y)]
            templates.append((target_dx, target_dy, mid_points))
        return templates

    def shortest_path_cost(self, board: Board, colour: Colour) -> float:
        """
        Dijkstra 算法计算最短路径。
        """
        size = board.size
        dist: List[List[float]] = [[float(INF)] * size for _ in range(size)]
        pq = []  # (cost, x, y)

        starts = self._identify_start_nodes(board, colour)
        for x, y, initial_cost in starts:
            node_w = self._get_node_cost(board, x, y, colour)
            if node_w < INF:
                total_cost = initial_cost + node_w
                if total_cost < dist[x][y]:
                    dist[x][y] = total_cost
                    heapq.heappush(pq, (total_cost, x, y))

        while pq:
            d, x, y = heapq.heappop(pq)
            if d > dist[x][y]: continue
            
            if self._is_at_target_edge(x, y, size, colour):
                return d

            self._expand_physical_neighbors(board, colour, x, y, d, dist, pq)
            self._expand_bridge_neighbors(board, colour, x, y, d, dist, pq)

        return float("inf")

    def evaluate_leaf(self, board: Board, colour: Colour) -> float:
        """
        基于 Dijkstra Cost 的局面评估。
        返回范围: [-1.0, 1.0] (负数代表劣势，正数代表优势)
        
        [Updated v8.0] 重构了评分公式以侧重自身连接。
        """
        my_cost = self.shortest_path_cost(board, colour)
        opp_cost = self.shortest_path_cost(board, Colour.opposite(colour))

        if my_cost >= INF and opp_cost >= INF: return 0.0 # 双死
        if my_cost >= INF: return -1.0 # 我方死路
        if opp_cost >= INF: return 1.0  # 敌方死路

        # [Key Change Here]
        # 旧公式: score = opp_cost - my_cost
        # 新公式: 强调降低 My Cost
        # 解释：
        # - My Cost 越小越好 (负号) -> 权重 1.2
        # - Opp Cost 越大越好 (正号) -> 权重 0.8
        # 这会引导 MCTS 选择那些 "稍微放过对手一点，但能显著缩短自己路径" 的点。
        score = (opp_cost * 0.8) - (my_cost * 1.2)
        
        # 归一化
        normalized = score / (board.size * 0.8) # 调整分母适应新权重
        
        return max(-1.0, min(1.0, normalized))

    # ==========================================
    # 内部逻辑方法
    # ==========================================

    def _identify_start_nodes(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        size = board.size
        starts = []
        
        if colour == Colour.RED:
            for y in range(size):
                starts.append((0, y, 0.0))
            # Edge Templates
            for y in range(size):
                n1_ok = self._is_safe(board, 0, y, colour)
                n2_ok = self._is_safe(board, 0, y+1, colour) if y+1 < size else False
                if n1_ok or n2_ok: starts.append((1, y, 0.01)) 
        else: # BLUE
            for x in range(size):
                starts.append((x, 0, 0.0))
            # Edge Templates
            for x in range(size):
                n1_ok = self._is_safe(board, x, 0, colour)
                n2_ok = self._is_safe(board, x+1, 0, colour) if x+1 < size else False
                if n1_ok or n2_ok: starts.append((x, 1, 0.01))
        return starts

    def _expand_physical_neighbors(self, board: Board, colour: Colour, 
                                   x: int, y: int, current_dist: float, 
                                   dist_map: List[List[float]], pq: List):
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx = x + Tile.I_DISPLACEMENTS[i]
            ny = y + Tile.J_DISPLACEMENTS[i]
            
            if self._is_valid(nx, ny, board.size):
                node_cost = self._get_node_cost(board, nx, ny, colour)
                if node_cost >= INF: continue
                
                new_dist = current_dist + node_cost
                if new_dist < dist_map[nx][ny]:
                    dist_map[nx][ny] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

    def _expand_bridge_neighbors(self, board: Board, colour: Colour, 
                                 x: int, y: int, current_dist: float, 
                                 dist_map: List[List[float]], pq: List):
        opp_colour = Colour.opposite(colour)
        size = board.size

        for t_dx, t_dy, mids in self.bridge_templates:
            tx, ty = x + t_dx, y + t_dy
            if not self._is_valid(tx, ty, size): continue
                
            bridge_intact = False
            for mx_off, my_off in mids:
                mx, my = x + mx_off, y + my_off
                if self._is_valid(mx, my, size):
                    if board.tiles[mx][my].colour != opp_colour:
                        bridge_intact = True
                        break
            
            if bridge_intact:
                node_cost = self._get_node_cost(board, tx, ty, colour)
                if node_cost >= INF: continue
                new_dist = current_dist + 0.01 + node_cost
                if new_dist < dist_map[tx][ty]:
                    dist_map[tx][ty] = new_dist
                    heapq.heappush(pq, (new_dist, tx, ty))

    def _get_node_cost(self, board: Board, x: int, y: int, colour: Colour) -> float:
        tile_colour = board.tiles[x][y].colour
        if tile_colour == colour: return 0.0
        if tile_colour is not None: return float(INF)
        base_cost = 1.0
        penalty = self._get_influence_penalty(board, x, y, Colour.opposite(colour))
        if penalty >= 10.0: return 100.0 
        return base_cost + penalty

    def _get_influence_penalty(self, board: Board, x: int, y: int, opp_colour: Colour) -> float:
        size = board.size
        penalty = 0.0
        opp_neighbors = 0
        neighbors_indices = []
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx = x + Tile.I_DISPLACEMENTS[i]
            ny = y + Tile.J_DISPLACEMENTS[i]
            if 0 <= nx < size and 0 <= ny < size:
                if board.tiles[nx][ny].colour == opp_colour:
                    opp_neighbors += 1
                    neighbors_indices.append(i)
        
        if opp_neighbors >= 2: penalty += 0.5 * opp_neighbors

        is_bridge_eye = False
        if len(neighbors_indices) >= 2:
            for idx1 in neighbors_indices:
                for idx2 in neighbors_indices:
                    if idx1 == idx2: continue
                    diff = abs(idx1 - idx2)
                    if diff > 3: diff = 6 - diff
                    if diff == 2:
                        is_bridge_eye = True
                        break
                if is_bridge_eye: break
        
        if is_bridge_eye: penalty += 2.0 
        return penalty

    def _is_safe(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        if not self._is_valid(x, y, board.size): return False
        c = board.tiles[x][y].colour
        return c == colour or c is None

    def _is_valid(self, x: int, y: int, size: int) -> bool:
        return 0 <= x < size and 0 <= y < size

    def _is_at_target_edge(self, x: int, y: int, size: int, colour: Colour) -> bool:
        if colour == Colour.RED: return x == size - 1
        return y == size - 1