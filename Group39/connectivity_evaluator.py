"""
connectivity_evaluator.py
[Strategy Enhanced Version]
基于 Dijkstra + 虚拟连接 (VC) + 动态势能场 (Potential Field) 的评估器。

核心特性：
1. 虚拟连接 (Virtual Connections):
   - 识别棋盘上的"双桥"结构。
   - 如果 A 和 B 构成双桥且中间未被阻断，则 A->B 的边权重为 0.01 (几乎连通)。
   
2. 动态势能场 (Dynamic Node Cost):
   - 空位不再平权。
   - 处于敌方"双桥眼"或被敌方包围的空位，通行代价显著增加 (High Penalty)。
   - 这使得 AI 能识别"死路"：即使物理上还有空位，但已被敌方 VC 封锁，Dijkstra 会绕道。

3. 边缘模板 (Edge Templates):
   - 不仅从物理边缘 (Row 0) 开始搜索。
   - 如果 Row 1 的点通过桥接关系"挂"在边缘上，也视为起点 (Cost ≈ 0)。
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
        原理：
        Hex 节点的 6 个邻居向量是循环相邻的。
        桥接点 T 是当前点 P 沿方向 i 走一步，再沿方向 i+1 走一步的结果。
        即 T = P + v[i] + v[i+1]。
        中间的两个"眼"分别是 P+v[i] 和 P+v[i+1]。
        """
        templates = []
        for i in range(Tile.NEIGHBOUR_COUNT):
            # 向量 v1
            v1_x, v1_y = Tile.I_DISPLACEMENTS[i], Tile.J_DISPLACEMENTS[i]
            
            # 向量 v2 (相邻方向)
            next_i = (i + 1) % Tile.NEIGHBOUR_COUNT
            v2_x, v2_y = Tile.I_DISPLACEMENTS[next_i], Tile.J_DISPLACEMENTS[next_i]
            
            # 目标位置 (向量和)
            target_dx = v1_x + v2_x
            target_dy = v1_y + v2_y
            
            # 两个中间眼位
            mid_points = [(v1_x, v1_y), (v2_x, v2_y)]
            
            templates.append((target_dx, target_dy, mid_points))
            
        return templates

    def shortest_path_cost(self, board: Board, colour: Colour) -> float:
        """
        Dijkstra 算法计算最短路径。
        
        Cost 定义：
        - 经过己方棋子: 0.0
        - 经过安全空位: 1.0
        - 经过敌方控制区空位: 1.0 + Penalty (势能场)
        - 经过虚拟连接边: 0.01 (VC)
        """
        size = board.size
        
        # dist[x][y] 记录从起点边到达 (x,y) 的最小代价
        dist: List[List[float]] = [[float(INF)] * size for _ in range(size)]
        pq = []  # (cost, x, y)

        # --- 1. 初始化起点 (包含物理边缘和边缘模板) ---
        starts = self._identify_start_nodes(board, colour)
        
        for x, y, initial_cost in starts:
            # 叠加节点的实际通行代价 (如果是空位)
            node_w = self._get_node_cost(board, x, y, colour)
            if node_w < INF:
                total_cost = initial_cost + node_w
                if total_cost < dist[x][y]:
                    dist[x][y] = total_cost
                    heapq.heappush(pq, (total_cost, x, y))

        # --- 2. Dijkstra 主循环 ---
        while pq:
            d, x, y = heapq.heappop(pq)

            # 懒惰删除：如果发现更短路径，跳过旧记录
            if d > dist[x][y]:
                continue
            
            # 检查是否到达终点边
            if self._is_at_target_edge(x, y, size, colour):
                return d

            # --- 3. 扩展邻居 ---
            
            # A) 扩展物理邻居 (Distance = 1 格)
            # 代价 = 边权重(0) + 目标节点通行代价
            self._expand_physical_neighbors(board, colour, x, y, d, dist, pq)

            # B) 扩展虚拟连接 (Distance = 2 格 / Bridge)
            # 代价 = 边权重(0.01) + 目标节点通行代价
            # 仅当 (x,y) 本身是己方棋子或空位时才可发起桥接
            self._expand_bridge_neighbors(board, colour, x, y, d, dist, pq)

        return float("inf")

    def evaluate_leaf(self, board: Board, colour: Colour) -> float:
        """
        基于 Dijkstra Cost 的局面评估。
        返回范围: [-1.0, 1.0] (负数代表劣势，正数代表优势)
        """
        my_cost = self.shortest_path_cost(board, colour)
        opp_cost = self.shortest_path_cost(board, Colour.opposite(colour))

        if my_cost >= INF and opp_cost >= INF:
            return 0.0 # 双死 (罕见)
        if my_cost >= INF:
            return -1.0 # 我方死路
        if opp_cost >= INF:
            return 1.0  # 敌方死路

        # 评分公式：Cost 越小越好
        # 基础分 = 敌方代价 - 我方代价
        score = opp_cost - my_cost
        
        # 归一化：除以棋盘尺寸的一半作为缩放因子
        # 我们希望细微的 VC 差异 (0.01) 也能体现，但不要被大的 Cost 淹没
        # 因此这里不做过度的非线性压缩
        normalized = score / (board.size * 0.6)
        
        return max(-1.0, min(1.0, normalized))

    # ==========================================
    # 内部逻辑方法
    # ==========================================

    def _identify_start_nodes(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        """
        识别所有可以作为起点的节点。
        包括：
        1. 物理边缘上的点 (Cost 0)
        2. [边缘模板] 距离边缘一步但通过桥接相连的点 (Cost 0.01)
        """
        size = board.size
        starts = []
        
        if colour == Colour.RED:
            # 目标：连接 Top(Row 0) -> Bottom
            # 1. 物理边缘 (Row 0)
            for y in range(size):
                starts.append((0, y, 0.0))
            
            # 2. 边缘模板 (Row 1)
            # 如果 (1, y) 能通过桥接到虚拟的 Row -1，则也视为起点
            # Hex 几何中，(1, y) 的上方邻居是 (0, y) 和 (0, y+1)
            # 它的"上方桥接"通常涉及到 (0, y) 和 (0, y+1) 的空位情况
            # 简化逻辑：如果 (1, y) 是己方棋子或空位，且 (0, y) 或 (0, y+1) 是空的/己方的
            # 我们给它一个极低的初始惩罚，视为"准起点"
            for y in range(size):
                # 检查上方两个邻居是否通畅
                n1_ok = self._is_safe(board, 0, y, colour)
                n2_ok = self._is_safe(board, 0, y+1, colour) if y+1 < size else False
                
                # 如果有两个支撑点，或者形成特定形状，视为 Edge Template
                if n1_ok or n2_ok:
                    starts.append((1, y, 0.01)) 

        else: # BLUE
            # 目标：连接 Left(Col 0) -> Right
            # 1. 物理边缘 (Col 0)
            for x in range(size):
                starts.append((x, 0, 0.0))
                
            # 2. 边缘模板 (Col 1)
            for x in range(size):
                n1_ok = self._is_safe(board, x, 0, colour)
                n2_ok = self._is_safe(board, x+1, 0, colour) if x+1 < size else False
                if n1_ok or n2_ok:
                    starts.append((x, 1, 0.01))

        return starts

    def _expand_physical_neighbors(self, board: Board, colour: Colour, 
                                   x: int, y: int, current_dist: float, 
                                   dist_map: List[List[float]], pq: List):
        """扩展 6 个物理相邻节点"""
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
        """
        扩展虚拟连接 (双桥)。
        如果中间路径畅通，则可以直接"跳"到目标点，边权重仅为 0.01。
        """
        opp_colour = Colour.opposite(colour)
        size = board.size

        for t_dx, t_dy, mids in self.bridge_templates:
            tx, ty = x + t_dx, y + t_dy
            
            if not self._is_valid(tx, ty, size):
                continue
                
            # 检查中间点是否阻断 (只要有一个中间点不是敌方，桥就是通的)
            # 更严格的大局观：如果中间点是空的，就是 Virtual Connection。
            # 如果中间点已经被我方占领，那其实是物理连接（已被上面的函数处理），这里再算一次也无妨。
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
                
                # VC Cost = 0.01 (几乎免费的跳跃)
                new_dist = current_dist + 0.01 + node_cost
                
                if new_dist < dist_map[tx][ty]:
                    dist_map[tx][ty] = new_dist
                    heapq.heappush(pq, (new_dist, tx, ty))

    def _get_node_cost(self, board: Board, x: int, y: int, colour: Colour) -> float:
        """
        计算节点的通行代价 (动态势能场)。
        - 己方棋子: 0.0
        - 敌方棋子: INF
        - 空位: 1.0 + Influence Penalty
        """
        tile_colour = board.tiles[x][y].colour
        
        if tile_colour == colour:
            return 0.0
        if tile_colour is not None: # 敌方棋子
            return float(INF)
            
        # --- 空位评估 (势能场核心) ---
        base_cost = 1.0
        penalty = self._get_influence_penalty(board, x, y, Colour.opposite(colour))
        
        # 如果 Penalty 极高 (死路)，我们可以认为 Cost 接近 INF，但为了 Dijkstra 连贯性，给一个大数
        if penalty >= 10.0:
            return 100.0 # 软阻断 (Soft Block)
            
        return base_cost + penalty

    def _get_influence_penalty(self, board: Board, x: int, y: int, opp_colour: Colour) -> float:
        """
        计算敌方对空位 (x,y) 的控制力/威胁度。
        如果 (x,y) 是敌方的双桥眼，或者是被敌方包围的死地，给予高惩罚。
        """
        size = board.size
        penalty = 0.0
        opp_neighbors = 0
        
        # 1. 检查物理包围 (Direct Neighbors)
        neighbors_indices = []
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx = x + Tile.I_DISPLACEMENTS[i]
            ny = y + Tile.J_DISPLACEMENTS[i]
            if 0 <= nx < size and 0 <= ny < size:
                if board.tiles[nx][ny].colour == opp_colour:
                    opp_neighbors += 1
                    neighbors_indices.append(i)
        
        # 如果周围有 >= 2 个敌人，且不构成连续防线，可能比较危险
        if opp_neighbors >= 2:
            penalty += 0.5 * opp_neighbors

        # 2. 检查双桥眼 (Bridge Eye Detection)
        # 如果 (x,y) 夹在两个不相邻的敌方棋子中间 (Diff=2)，说明它是双桥的眼
        # 填这个眼通常是低效的 (除非为了 Save Bridge，但这里是计算通行代价)
        # 对于通行者来说，这个点被敌方双向锁定，通过它并不能切断敌方，反而自己容易被吃。
        
        # 简化版双桥眼检测
        is_bridge_eye = False
        if len(neighbors_indices) >= 2:
            for idx1 in neighbors_indices:
                for idx2 in neighbors_indices:
                    if idx1 == idx2: continue
                    diff = abs(idx1 - idx2)
                    if diff > 3: diff = 6 - diff
                    if diff == 2: # 马步关系
                        is_bridge_eye = True
                        break
                if is_bridge_eye: break
        
        if is_bridge_eye:
            # 这是一个极度危险的点。
            # 如果我方下在这里，对手一定会下另一个眼。
            # 对于连通性来说，这是一条极其昂贵的路。
            penalty += 2.0 

        return penalty

    def _is_safe(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """简单的安全性检查：未越界且不是敌方"""
        if not self._is_valid(x, y, board.size): return False
        c = board.tiles[x][y].colour
        return c == colour or c is None

    def _is_valid(self, x: int, y: int, size: int) -> bool:
        return 0 <= x < size and 0 <= y < size

    def _is_at_target_edge(self, x: int, y: int, size: int, colour: Colour) -> bool:
        if colour == Colour.RED:
            return x == size - 1
        else:
            return y == size - 1