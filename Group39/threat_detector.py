"""
threat_detector.py
[Strategy Enhanced Version]
威胁检测模块。

核心升级：
1. 集成 "Useless Cut" (无效切断) 过滤：
   - 在检测 LOSE 威胁和 Blocking 模式时，自动忽略对手双桥的"眼"。
   - 防止 AI 在必败或防守时走出"填眼"的废棋。
2. 纯几何模式识别：
   - 不依赖 BFS，使用向量计算快速识别 Bridge 和 Connection。
"""

import copy
from typing import List, Tuple, Set

from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile

class ThreatDetector:
    """
    威胁检测器
    
    职责：
    1. Detect Immediate Threats (Win/Lose in 1 move).
    2. Detect Simple Patterns (Bridge, Connection, Blocking) for fallback/heuristics.
    """
    
    def __init__(self, colour: Colour):
        self.colour = colour
    
    def detect_immediate_threats(self, board: Board, opponent_colour: Colour) -> List[Tuple[int, int, str]]:
        """
        检测一手输/赢的情况。
        优先级：WIN (我方必胜) > LOSE (对手必胜/我方必败)。
        """
        threats = []
        
        # --- 1. 检查我方 WIN ---
        # 优化：只检查靠近我方棋子或目标底边的空位
        win_candidates = self._get_candidates(board, self.colour)
        for x, y in win_candidates:
            if self._quick_win_check(board, x, y, self.colour):
                # 发现必胜，直接返回 (最高优先级)
                return [(x, y, "WIN")]
        
        # --- 2. 检查对手 WIN (即我方 LOSE) ---
        lose_candidates = self._get_candidates(board, opponent_colour)
        for x, y in lose_candidates:
            # [关键过滤] 
            # 如果 (x,y) 是对手双桥的眼，填了它并不能阻止对手连接 (对手会走另一个眼)。
            # 因此，不要将其标记为"可防守的威胁"，避免 AI 浪费一手棋去填眼。
            if self._is_useless_cut_simple(board, x, y, self.colour):
                continue

            if self._quick_win_check(board, x, y, opponent_colour):
                threats.append((x, y, "LOSE"))
        
        return threats

    def detect_simple_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        """
        检测几何模式，返回高价值点列表。
        格式: [(x, y, weight), ...]
        """
        patterns = []
        
        # 1. 桥接模式 (Bridge) - 权重 0.8
        # 连接我方两个不相邻的棋子
        bridges = self._detect_bridge_geometrically(board, colour)
        for x, y in bridges:
            patterns.append((x, y, 0.8))
            
        # 2. 边缘连接模式 (Connection) - 权重 0.6
        # 帮助边缘棋子建立连接
        connections = self._detect_edge_connection_geometrically(board, colour)
        for x, y in connections:
            patterns.append((x, y, 0.6))
            
        # 3. 阻断模式 (Blocking) - 权重 0.7
        # 寻找对手的桥接点并进行阻断
        # [关键] 必须过滤掉无效切断
        opp = Colour.opposite(colour)
        opp_bridges = self._detect_bridge_geometrically(board, opp)
        
        for x, y in opp_bridges:
            # 只有当切断是有效的 (即不是双桥的眼) 时，才推荐阻断
            if not self._is_useless_cut_simple(board, x, y, colour):
                patterns.append((x, y, 0.7))

        return patterns

    # ==========================================
    # 核心辅助逻辑
    # ==========================================

    def _get_candidates(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        获取可能形成连接的关键候选点 (性能优化)。
        只返回：
        1. 目标底边上的空位。
        2. 已有同色棋子周围 2 格范围内的空位。
        """
        candidates = set()
        size = board.size
        
        # 1. 边界种子
        if colour == Colour.RED:
            # Red 关注 Top (Row 0) and Bottom (Row size-1)
            for y in range(size):
                if board.tiles[0][y].colour is None: candidates.add((0, y))
                if board.tiles[size-1][y].colour is None: candidates.add((size-1, y))
        else:
            # Blue 关注 Left (Col 0) and Right (Col size-1)
            for x in range(size):
                if board.tiles[x][0].colour is None: candidates.add((x, 0))
                if board.tiles[x][size-1].colour is None: candidates.add((x, size-1))

        # 2. 棋子周边 (Radius 2)
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == colour:
                    # 检查 2 格半径
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                if board.tiles[nx][ny].colour is None:
                                    candidates.add((nx, ny))
        return list(candidates)
    
    def _quick_win_check(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        快速模拟落子并检查是否胜利。
        使用 Board 内置的 has_ended (BFS/DFS) 检查物理连接。
        """
        # 1. 临时落子
        original_colour = board.tiles[x][y].colour
        board.tiles[x][y].colour = colour
        board._winner = None # 清除缓存
        
        # 2. 检查胜利
        has_won = board.has_ended(colour)
        
        # 3. 恢复状态
        board.tiles[x][y].colour = original_colour
        board._winner = None
        
        return has_won

    def _is_useless_cut_simple(self, board: Board, x: int, y: int, my_colour: Colour) -> bool:
        """
        [大局观核心] 简版无效切断检测 (用于 ThreatDetector 独立运行)。
        
        判断 (x,y) 是否是对手双桥 (Two-Bridge) 的其中一个眼。
        逻辑：
        如果 (x,y) 被对手两颗棋子 A, B 以马步 (Diff=2) 夹击，
        且 A, B 之间的另一条路 Q (另一个眼) 是空的。
        那么切断 (x,y) 是无效的，因为对手可以走 Q。
        """
        opp_colour = Colour.opposite(my_colour)
        size = board.size
        
        # 遍历所有 6 个方向
        for i in range(Tile.NEIGHBOUR_COUNT):
            # A 在方向 i
            ax, ay = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
            
            # B 在方向 i+2 (马步相对)
            i2 = (i + 2) % Tile.NEIGHBOUR_COUNT
            bx, by = x + Tile.I_DISPLACEMENTS[i2], y + Tile.J_DISPLACEMENTS[i2]
            
            # 检查 A, B 是否为对手棋子
            if not (self._is_colour(board, ax, ay, opp_colour) and 
                    self._is_colour(board, bx, by, opp_colour)):
                continue
                
            # 找到了夹击，检查双桥的另一个眼 Q (方向 i+1)
            im = (i + 1) % Tile.NEIGHBOUR_COUNT
            qx, qy = x + Tile.I_DISPLACEMENTS[im], y + Tile.J_DISPLACEMENTS[im]
            
            # 如果 Q 是空的，说明双桥结构完整，(x,y) 只是其中一个眼
            if 0 <= qx < size and 0 <= qy < size and board.tiles[qx][qy].colour is None:
                return True # 切断无效
                
        return False

    def _detect_bridge_geometrically(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        纯几何方式检测桥接点 (不依赖 BFS)。
        返回所有能连接两个不相邻同色棋子的空位。
        """
        bridges = []
        size = board.size
        
        # 扫描所有空位
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is not None: continue
                
                # 获取周围同色棋子的方向索引
                neighbors_idx = []
                for i in range(Tile.NEIGHBOUR_COUNT):
                    nx, ny = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
                    if self._is_colour(board, nx, ny, colour):
                        neighbors_idx.append(i)
                
                if len(neighbors_idx) < 2: continue
                
                # 检查是否存在不相邻的邻居 (Diff >= 2)
                # Diff=1 是紧挨着 (Solid Connection)，不是 Bridge
                is_bridge = False
                for idx1 in range(len(neighbors_idx)):
                    for idx2 in range(idx1 + 1, len(neighbors_idx)):
                        diff = abs(neighbors_idx[idx1] - neighbors_idx[idx2])
                        if diff > 3: diff = 6 - diff
                        
                        if diff >= 2: # 只要不是紧挨着，就算某种弱连接/桥
                            is_bridge = True
                            break
                    if is_bridge: break
                
                if is_bridge:
                    bridges.append((x, y))
                    
        return bridges

    def _detect_edge_connection_geometrically(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """
        几何检测边缘连接潜力。
        寻找位于二路、三路且能连接到己方棋子的空位。
        """
        points = []
        size = board.size
        
        # 定义关注的边缘区域 (Top/Bottom or Left/Right)
        if colour == Colour.RED:
            # 关注 Row 0, 1 (Top) 和 Row size-2, size-1 (Bottom)
            target_rows = [0, 1, size-2, size-1]
            for r in target_rows:
                if 0 <= r < size:
                    for c in range(size):
                        if board.tiles[r][c].colour is None:
                            # 检查附近有没有我方棋子 (提供支撑)
                            if self._has_neighbor(board, r, c, colour):
                                points.append((r, c))
        else:
            # 关注 Col 0, 1 (Left) 和 Col size-2, size-1 (Right)
            target_cols = [0, 1, size-2, size-1]
            for c in target_cols:
                if 0 <= c < size:
                    for r in range(size):
                        if board.tiles[r][c].colour is None:
                            if self._has_neighbor(board, r, c, colour):
                                points.append((r, c))
        return points

    def _is_colour(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        if not (0 <= x < board.size and 0 <= y < board.size): return False
        return board.tiles[x][y].colour == colour

    def _has_neighbor(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx, ny = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
            if self._is_colour(board, nx, ny, colour): return True
        return False