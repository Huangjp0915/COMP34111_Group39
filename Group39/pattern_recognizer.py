"""
pattern_recognizer.py
[Strategy Enhanced Version]
基于 MoHex 风格 Gamma 权重的模式识别器。

核心架构：
1. Inferior Cells: 无效切断/死子 -> Gamma 极小 (1e-5).
2. Local Patterns: 针对对手上一手的应手 (Cap/Save) -> Gamma 极大 (100.0+).
3. Global Patterns: 自身结构延伸 (Bridge/Edge) -> Gamma 中等 (10.0).
4. Default: 未知区域 -> Gamma 基准 (1.0).

这种架构确保了"战术急所" (Local) 永远优先于 "战略大场" (Global)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile
from .utils import hash_board

class PatternRecognizer:
    
    # --- MoHex 风格的 Gamma 权重配置 ---
    GAMMA_INFERIOR = 1e-5    # 劣势点 (Prunable)
    GAMMA_DEFAULT = 1.0      # 默认点
    
    # Global 权重 (战略)
    GAMMA_GLOBAL_BRIDGE = 15.0   # 主动双桥延伸
    GAMMA_GLOBAL_EDGE = 12.0     # 边缘模板
    GAMMA_GLOBAL_CONN = 5.0      # 普通连接
    
    # Local 权重 (战术，必须压倒 Global)
    GAMMA_LOCAL_URGENT = 1000.0  # 救命/必应 (Save Bridge)
    GAMMA_LOCAL_CAP = 200.0      # 镇头/强烈反击 (Cap)
    GAMMA_LOCAL_SHOULDER = 80.0  # 肩冲

    def __init__(self, colour: Colour):
        self.colour = colour
        # 缓存: (board_hash, opp_last_move_idx) -> {move_idx: gamma}
        self._cache: Dict[Tuple[str, int], Dict[Tuple[int, int], float]] = {}
        self._board_hash_cache: Dict[int, str] = {}

    def get_all_gammas(self, board: Board, valid_moves: List[Move], opp_last_move: Optional[Move]) -> Dict[Tuple[int, int], float]:
        """
        计算所有合法走法的 Gamma 值。
        MCTS 使用此函数生成 Prior 概率分布。
        
        Args:
            valid_moves: 待评估的空位列表
            opp_last_move: 对手上一手 (用于激活 Local Pattern)
        Returns:
            {(x,y): gamma_value}
        """
        # 1. 缓存键生成
        bid = id(board)
        b_hash = self._board_hash_cache.get(bid)
        if b_hash is None:
            b_hash = hash_board(board)
            if len(self._board_hash_cache) > 256: self._board_hash_cache.clear()
            self._board_hash_cache[bid] = b_hash

        opp_idx = -1
        if opp_last_move:
            opp_idx = opp_last_move.x * board.size + opp_last_move.y
            
        cache_key = (b_hash, opp_idx)
        if cache_key in self._cache:
            # 过滤掉不在 valid_moves 里的缓存项 (虽然通常是一致的)
            full_cache = self._cache[cache_key]
            return { (m.x, m.y): full_cache.get((m.x, m.y), self.GAMMA_DEFAULT) for m in valid_moves }

        gammas = {}
        
        # 2. 遍历所有合法点计算 Gamma
        for move in valid_moves:
            x, y = move.x, move.y
            
            # A. Inferior Cell Check (第一优先级：剪枝)
            if self._is_inferior(board, x, y, self.colour):
                gammas[(x, y)] = self.GAMMA_INFERIOR
                continue
            
            # B. Local Pattern (第二优先级：应手)
            local_g = 0.0
            if opp_last_move:
                local_g = self._eval_local_pattern(board, x, y, self.colour, opp_last_move)
            
            # 如果发现了紧急 Local Pattern，直接赋予极高权重
            # MoHex 论文中是 sum(local, global)，但这里我们用 Max 机制确保 Local 不被淹没
            if local_g >= self.GAMMA_LOCAL_SHOULDER:
                gammas[(x, y)] = local_g
                continue
                
            # C. Global Pattern (第三优先级：大场)
            global_g = self._eval_global_pattern(board, x, y, self.colour)
            
            # 融合：Local 的微小加成 + Global + Base
            # 注意：如果 local_g 很小 (比如只是靠近对手但没威胁)，它只作为微调
            final_gamma = max(self.GAMMA_DEFAULT, global_g) + local_g
            gammas[(x, y)] = final_gamma

        self._cache[cache_key] = gammas
        return gammas

    def get_prior(self, board: Board, move: Move, colour: Colour) -> float:
        """
        [兼容接口] 获取单点的归一化评分 (0.0 - 1.0)。
        用于 Agent 的 _score_move_light 后处理。
        注意：此处没有 opp_last_move 上下文，只能计算 Global + Inferior。
        """
        # 检查劣势点
        if self._is_inferior(board, move.x, move.y, colour):
            return 0.0
            
        # 计算 Global Gamma
        gamma = self._eval_global_pattern(board, move.x, move.y, colour)
        
        # 简单归一化映射 (1.0 -> 0.1, 15.0 -> 0.8)
        if gamma >= self.GAMMA_GLOBAL_BRIDGE: return 0.8
        if gamma >= self.GAMMA_GLOBAL_EDGE: return 0.6
        return 0.1

    def detect_simple_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        """
        [兼容接口] 返回高价值候选点列表 (用于 Fallback)。
        """
        candidates = []
        # 简单扫描全盘空位 (或者只扫描有邻居的)
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    # 只要不是劣势点，且有一定 Global 价值
                    if not self._is_inferior(board, x, y, colour):
                        g = self._eval_global_pattern(board, x, y, colour)
                        if g > self.GAMMA_DEFAULT:
                            # 归一化权重
                            w = min(0.9, g / 20.0)
                            candidates.append((x, y, w))
        
        candidates.sort(key=lambda t: t[2], reverse=True)
        return candidates

    # ==========================================
    # 1. Inferior Engine (劣势点判定)
    # ==========================================
    
    def _is_inferior(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        判断是否为劣势点 (Useless Cut / Dead Cell)。
        """
        # 1. 无效切断对手双桥
        if self._is_useless_cut(board, x, y, colour):
            return True
            
        return False

    def _is_useless_cut(self, board: Board, x: int, y: int, my_colour: Colour) -> bool:
        """
        对手双桥的眼 = 劣势点。
        如果我方下在这里，对手只要下另一个眼就能连通。
        """
        opp = Colour.opposite(my_colour)
        size = board.size
        
        # 检查是否夹在对手马步中间
        for i in range(Tile.NEIGHBOUR_COUNT):
            # A 在方向 i
            ax, ay = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
            
            # B 在方向 i+2 (马步)
            i2 = (i + 2) % Tile.NEIGHBOUR_COUNT
            bx, by = x + Tile.I_DISPLACEMENTS[i2], y + Tile.J_DISPLACEMENTS[i2]
            
            # 检查 A, B 是否为对手
            if self._is_co(board, ax, ay, opp) and self._is_co(board, bx, by, opp):
                # 检查另一个眼 Q (方向 i+1)
                im = (i + 1) % Tile.NEIGHBOUR_COUNT
                qx, qy = x + Tile.I_DISPLACEMENTS[im], y + Tile.J_DISPLACEMENTS[im]
                
                # 如果 Q 是空的，说明对手有双保险
                if self._is_empty(board, qx, qy):
                    return True
        return False

    # ==========================================
    # 2. Local Pattern Engine (局部应手)
    # ==========================================
    
    def _eval_local_pattern(self, board: Board, x: int, y: int, colour: Colour, opp_move: Move) -> float:
        """
        评估针对 opp_move 的局部价值。
        """
        # 距离过滤：如果离对手上一手太远 (>2格)，通常不是 Local Pattern
        dist = abs(x - opp_move.x) + abs(y - opp_move.y)
        if dist > 3: return 0.0
        
        val = 0.0
        
        # 1. Save Bridge (救命) - 最高级
        # 如果对手这一手威胁到了我的双桥，必须立刻补
        if self._is_save_bridge(board, x, y, colour, opp_move):
            return self.GAMMA_LOCAL_URGENT
            
        # 2. The Cap (镇头) - 强力压制
        # 检查 (x,y) 是否在 opp_move 的进攻方向前方
        if self._is_cap_move(board, x, y, colour, opp_move):
            val = max(val, self.GAMMA_LOCAL_CAP)
            
        # 3. Shoulder Hit (肩冲) - 侧面压制
        # 如果 (x,y) 和 opp_move 紧邻，且位于其侧面
        if dist <= 2: 
            # 简单检查：如果 (x,y) 的邻居里有 opp_move
            if self._is_neighbor(x, y, opp_move.x, opp_move.y):
                val = max(val, self.GAMMA_LOCAL_SHOULDER)
            
        return val

    def _is_save_bridge(self, board: Board, x: int, y: int, colour: Colour, opp_move: Move) -> bool:
        """
        检测：对手刚下的 opp_move 是否切了我双桥的一只眼，而 (x,y) 是另一只眼。
        """
        # 检查 (x,y) 和 opp_move 是否是兄弟眼 (即关于两个我方棋子对称)
        # 简易判定：(x,y) 和 opp_move 必须都是空的(但在对手下之前)，且它们共同邻接两个我方棋子
        
        # 1. (x,y) 必须邻接 opp_move 吗？不一定，但在 Hex 中眼通常是邻居 (distance 1)
        if not self._is_neighbor(x, y, opp_move.x, opp_move.y):
            return False
            
        # 2. 检查 (x,y) 和 opp_move 是否共同邻接两个我方棋子
        my_shared = 0
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx, ny = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
            # 这是一个共同邻居吗？
            if self._is_co(board, nx, ny, colour):
                if self._is_neighbor(nx, ny, opp_move.x, opp_move.y):
                    my_shared += 1
        
        return my_shared >= 2

    def _is_cap_move(self, board: Board, x: int, y: int, colour: Colour, opp_move: Move) -> bool:
        """
        判断 (x,y) 是否是针对 opp_move 的镇头 (Cap)。
        定义：位于对手进攻方向的正前方。
        """
        opp_c = Colour.opposite(colour)
        
        # 计算相对位置
        dx = x - opp_move.x
        dy = y - opp_move.y
        
        # 根据对手颜色判断进攻方向
        if opp_c == Colour.RED: 
            # RED 目标 Row 增加 (Top -> Bottom)
            # 有效阻挡应该在 x > opp.x
            # (1, 0) 是正下方，(0, 1) 是右下方
            # (1, -1) 是左下方 (Knight move)
            if dx == 1 and dy == 0: return True  # Direct blocking
            if dx == 0 and dy == 1: return True  # Direct blocking
            if dx == 1 and dy == -1: return True # Knight Cap
            
        else: # BLUE
            # BLUE 目标 Col 增加 (Left -> Right)
            # 有效阻挡 y > opp.y
            if dy == 1: return True
            if dx == -1 and dy == 1: return True
            
        return False

    # ==========================================
    # 3. Global Pattern Engine (全局结构)
    # ==========================================
    
    def _eval_global_pattern(self, board: Board, x: int, y: int, colour: Colour) -> float:
        val = 0.0
        
        # 1. 边缘模板
        if self._is_edge_template(board, x, y, colour):
            val = max(val, self.GAMMA_GLOBAL_EDGE)
            
        # 2. 主动双桥延伸
        if self._detect_bridge_extension(board, x, y, colour):
            val = max(val, self.GAMMA_GLOBAL_BRIDGE)
            
        return val

    def _detect_bridge_extension(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        检测 (x,y) 是否能与现有我方棋子形成"进攻性"双桥。
        要求：
        1. 与我方某子 P 构成马步。
        2. 中间两个眼是空的。
        3. 方向是朝向目标边的 (Forward)。
        """
        size = board.size
        
        # 遍历所有可能的马步源头 P
        # P = (x,y) - (v1 + v2) => 反向查找比较麻烦
        # 不如遍历 P = (x,y) + (v1 + v2)，看 P 是否是我方棋子
        # 这样 (x,y) 和 P 构成双桥，且如果 (x,y) 更靠前，则 P 是源头
        
        for i in range(Tile.NEIGHBOUR_COUNT):
            v1x, v1y = Tile.I_DISPLACEMENTS[i], Tile.J_DISPLACEMENTS[i]
            next_i = (i + 1) % Tile.NEIGHBOUR_COUNT
            v2x, v2y = Tile.I_DISPLACEMENTS[next_i], Tile.J_DISPLACEMENTS[next_i]
            
            # P 是潜在的我方棋子
            px, py = x + v1x + v2x, y + v1y + v2y
            
            if self._is_co(board, px, py, colour):
                # 检查眼
                m1x, m1y = x + v1x, y + v1y
                m2x, m2y = x + v2x, y + v2y
                
                if self._is_empty(board, m1x, m1y) and self._is_empty(board, m2x, m2y):
                    # 检查方向性 (Forward check)
                    # (x,y) 必须比 P 更靠近目标边
                    is_forward = False
                    if colour == Colour.RED: 
                        # Target Row N (Max x). So x should be > px
                        if x > px: is_forward = True
                    else: 
                        # Target Col N (Max y). So y should be > py
                        if y > py: is_forward = True
                        
                    if is_forward:
                        return True
        return False

    def _is_edge_template(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        二路点检测 (Edge Template Base)。
        位于边缘第二线，且有潜力连接到边缘。
        """
        size = board.size
        if colour == Colour.RED:
            # Row 1 or Row size-2
            return x == 1 or x == size - 2
        else:
            # Col 1 or Col size-2
            return y == 1 or y == size - 2

    # ==========================================
    # Utils
    # ==========================================
    
    def _is_co(self, board: Board, x: int, y: int, c: Colour) -> bool:
        return 0 <= x < board.size and 0 <= y < board.size and board.tiles[x][y].colour == c
        
    def _is_empty(self, board: Board, x: int, y: int) -> bool:
        return 0 <= x < board.size and 0 <= y < board.size and board.tiles[x][y].colour is None
        
    def _is_neighbor(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx = x1 + Tile.I_DISPLACEMENTS[i]
            ny = y1 + Tile.J_DISPLACEMENTS[i]
            if nx == x2 and ny == y2: return True
        return False

    def _has_neighbor_of_colour(self, board: Board, x: int, y: int, target: Colour) -> bool:
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx, ny = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
            if self._is_co(board, nx, ny, target): return True
        return False
    
    def _get_relevant_empty_tiles(self, board: Board) -> List[Tuple[int, int]]:
        """获取棋子周围的空位"""
        relevant = set()
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is not None:
                    for i in range(Tile.NEIGHBOUR_COUNT):
                        nx, ny = x + Tile.I_DISPLACEMENTS[i], y + Tile.J_DISPLACEMENTS[i]
                        if self._is_empty(board, nx, ny):
                            relevant.add((nx, ny))
        return list(relevant)