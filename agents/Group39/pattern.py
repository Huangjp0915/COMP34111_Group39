"""
模式识别器 (优化修正版)
"""
from typing import List, Tuple, Optional
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile


class PatternRecognizer:
    """
    基于几何模板的快速模式识别器

    核心改进：
    1. O(1) 静态几何判定，移除 deepcopy
    2. 引入 Two-Bridge 显式模板 (Form vs Save)
    3. 区分 '连接两个分离组' 和 '简单延伸'
    """

    def __init__(self, colour: Colour):
        self.colour = colour
        # 预计算 Hex 两个相邻向量相加的 "对角/马步" 偏移量
        # 用于快速定位 Two-Bridge 的远端
        # key: direction_index, value: (dx, dy) representing vec[i] + vec[i+1]
        self.two_bridge_offsets = {}
        for i in range(Tile.NEIGHBOUR_COUNT):
            next_i = (i + 1) % Tile.NEIGHBOUR_COUNT
            dx = Tile.I_DISPLACEMENTS[i] + Tile.I_DISPLACEMENTS[next_i]
            dy = Tile.J_DISPLACEMENTS[i] + Tile.J_DISPLACEMENTS[next_i]
            self.two_bridge_offsets[i] = (dx, dy)

    def get_prior(self, board: Board, move: Move, colour: Colour) -> float:
        """
        计算单点先验概率 (Prior)
        范围: 0.0 ~ 1.0 (大部分点应在 0.1~0.4，极少数关键点 >0.8)
        """
        x, y = move.x, move.y
        prior = 0.1  # 基础分

        opp_colour = Colour.opposite(colour)

        # --- 1. Two-Bridge 几何检测 (最精准的形状) ---
        # 检查是否形成或挽救 Two-Bridge
        tb_status = self._check_two_bridge_geometry(board, x, y, colour)
        if tb_status == "SAVE":
            # 救命点：远端是我方，另一路连接点已被敌方占据 -> 必须占领此点
            return 0.95
        elif tb_status == "FORM":
            # 好点：远端是我方，另一路为空 -> 形成双保险连接
            prior += 0.35

        # --- 2. 传统桥接 (连接两个不相邻的友军) ---
        # 如果不是标准的 Two-Bridge，但连接了两个如果不走这步就断开的组
        # 判定标准：邻居中有 >=2 个友军，且它们在局部不连通 (diff >= 2 且中间没有友军)
        if self._is_general_connector(board, x, y, colour):
            prior += 0.25

        # --- 3. 接触战 (Contact Fight) ---
        # 只有当对手棋子就在旁边时，才给予少量关注 (避免过度泛化)
        if self._has_neighbor_of_colour(board, x, y, opp_colour):
            prior += 0.1  # 仅加 0.1，避免把所有接触点都当必争点

        # --- 4. 边缘连接 (Edge Extension) ---
        # 简单的 heuristic: 靠近我方目标底边的点略微加分
        if self._is_near_target_edge(board, x, y, colour):
            prior += 0.15

        return min(1.0, prior)

    def detect_simple_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        """
        (兼容旧接口) 返回一批推荐点，用于 Fallback
        """
        candidates = []
        # 只扫描有邻居的空位 (大幅减少计算量)
        relevant_empty = self._get_relevant_empty_tiles(board)

        for x, y in relevant_empty:
            move = Move(x, y)
            score = self.get_prior(board, move, colour)
            if score > 0.3:  # 只返回有价值的点
                candidates.append((x, y, score))

        return candidates

    def _check_two_bridge_geometry(self, board: Board, x: int, y: int, colour: Colour) -> Optional[str]:
        """
        检测 (x,y) 是否处于 Two-Bridge 结构的关键位

        结构定义：
        我的棋子 A --- 空位 P(当前) --- 我的棋子 B
                    |
                 空位/敌方 Q

        几何关系：A 和 B 的坐标差等于 (vec_i + vec_{i+1})
        """
        opp_colour = Colour.opposite(colour)
        size = board.size

        # 遍历 6 个方向对 (i, i+1)
        for i in range(Tile.NEIGHBOUR_COUNT):
            # 1. 检查方向 i 的邻居 (A)
            dx1, dy1 = Tile.I_DISPLACEMENTS[i], Tile.J_DISPLACEMENTS[i]
            nx1, ny1 = x + dx1, y + dy1

            if not (0 <= nx1 < size and 0 <= ny1 < size):
                continue

            # 如果方向 i 不是我方棋子，这个方向构不成以 (x,y) 为支点的 Two-Bridge
            if board.tiles[nx1][ny1].colour != colour:
                continue

            # 2. 检查 "远端" (B) -> (vec_i + vec_{i+1})
            # 注意：在 Hex 网格中，Two-bridge 的另一端 B 位于 (x,y) + vec_i + vec_{i+1} 吗？
            # 不，那是 "小三角"。
            # 让我们画图：
            # A (nx1, ny1) 是 (x,y) 的邻居 i。
            # 我们要找 B，使得 A-P-B 和 A-Q-B 形成菱形。
            # 在 Hex 中，如果 A 是邻居 i，B 应该是邻居 (i+1) 的方向... 不对。
            # 正确的 Two-Bridge 几何是：
            # P(x,y) 和 Q 是两个通过点。A 和 B 是被连接点。
            # 如果 P 是空位，A 是邻居 i。
            # B 应该是 "邻居 i+1 的邻居 i" 这种位置？
            #
            # 简化模型 (基于 Critique 建议的 v_i + v_{i+1})：
            # 如果我方棋子在 "远端" (x + vec_i + vec_{i+1})，
            # 那么连接点是 (x + vec_i) 和 (x + vec_{i+1})。
            #
            # 现在的视角：我是空位 P(x,y)。
            # 如果我处在 Two-Bridge 中，那么我的两个邻居 A 和 B 应该分别是 vec_i 和 vec_{i+1} 吗？
            # 如果 A 在 vec_i, B 在 vec_{i+1} (相邻方向)，那 A和B 是相邻的！这不需要桥。
            #
            # 修正几何理解：
            # Two-Bridge 的 A 和 B 是隔开的（马步）。
            # 比如 A 在 (0,0), B 在 (1, -2)。
            # 连接点是 (1, -1) 和 (0, -1)。
            # 对 (1, -1) 来说，A 是邻居，B 也是邻居。
            # 且 A 和 B 在 (1, -1) 的视角里，方向索引差为 2 (例如 dir 0 和 dir 2)。
            # 且中间那个方向 (dir 1) 的邻居 Q 也是连接点。

            # 所以逻辑是：
            # 检查我的邻居 j 和邻居 j+2 (索引差2)。
            # 如果邻居 j 是我方，邻居 j+2 是我方。
            # 那么我(P) 和 邻居 j+1 (Q) 就是这一对的双连接点。

            # 让我们用这个逻辑 (Diff=2 Check + Q Check)：
            next_i = (i + 1) % Tile.NEIGHBOUR_COUNT
            next_next_i = (i + 2) % Tile.NEIGHBOUR_COUNT

            # A 在方向 i
            # B 在方向 i+2 (相对 P)
            # Q 在方向 i+1 (相对 P)

            # 我们已知 A (nx1, ny1) 是 colour (上面 check 过)

            # 检查 B (方向 i+2)
            dx2, dy2 = Tile.I_DISPLACEMENTS[next_next_i], Tile.J_DISPLACEMENTS[next_next_i]
            nx2, ny2 = x + dx2, y + dy2

            if not (0 <= nx2 < size and 0 <= ny2 < size):
                continue

            if board.tiles[nx2][ny2].colour == colour:
                # 找到了 A 和 B，它们夹着 Q (方向 i+1) 和 P (我自己)
                # 检查 Q 的状态
                qx, qy = x + Tile.I_DISPLACEMENTS[next_i], y + Tile.J_DISPLACEMENTS[next_i]

                # 如果 Q 出界了，这不是完整的 bridge
                if not (0 <= qx < size and 0 <= qy < size):
                    continue

                q_tile = board.tiles[qx][qy]

                if q_tile.colour == opp_colour:
                    # Q 被敌人占了 -> 我必须占 P -> SAVE BRIDGE
                    return "SAVE"
                elif q_tile.colour is None:
                    # Q 是空的 -> 我占 P 就形成双保险 -> FORM BRIDGE
                    return "FORM"

        return None

    def _is_general_connector(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        通用连接检测：
        检查 (x,y) 的邻居中是否有两个友军，它们在局部是不连通的（Diff != 1）。
        如果是，那么 (x,y) 起到了连接作用。
        """
        my_neighbors_indices = []
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx = x + Tile.I_DISPLACEMENTS[i]
            ny = y + Tile.J_DISPLACEMENTS[i]
            if 0 <= nx < board.size and 0 <= ny < board.size:
                if board.tiles[nx][ny].colour == colour:
                    my_neighbors_indices.append(i)

        if len(my_neighbors_indices) < 2:
            return False

        # 检查是否存在不相邻的邻居 (Diff != 1 且 Diff != 5)
        # 如果所有邻居都是连在一起的 (e.g. 0, 1, 2)，那它们已经是一个团块了，填中间没意义。
        # 如果有 0 和 2，且 1 不是友军，那么 (x,y) 填补了 0-2 的缝隙。

        # 简单算法：对索引排序，看是否有间隔
        # 如果是 [0, 1]，间隔是 1 (连通)
        # 如果是 [0, 2]，间隔是 2 (不连通) -> 是 Connector
        # 如果是 [0, 1, 3]，0-1连通，1-3不连通 -> 是 Connector

        # 特殊情况：环形 0 和 5 是连通的

        # 只要找到一对邻居，它们不是“紧挨着”的，且中间那个位置没被我方占据
        # (其实 Two-Bridge check 已经覆盖了 diff=2 的情况，这里主要兜底 diff=3 对冲等情况)

        for idx1 in range(len(my_neighbors_indices)):
            for idx2 in range(idx1 + 1, len(my_neighbors_indices)):
                n1 = my_neighbors_indices[idx1]
                n2 = my_neighbors_indices[idx2]

                diff = abs(n1 - n2)
                if diff > 3: diff = 6 - diff

                if diff >= 2:
                    return True
        return False

    def _has_neighbor_of_colour(self, board: Board, x: int, y: int, target_colour: Colour) -> bool:
        """检查是否有特定颜色的邻居"""
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx = x + Tile.I_DISPLACEMENTS[i]
            ny = y + Tile.J_DISPLACEMENTS[i]
            if 0 <= nx < board.size and 0 <= ny < board.size:
                if board.tiles[nx][ny].colour == target_colour:
                    return True
        return False

    def _is_near_target_edge(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """检查是否靠近目标底边 (距离 <= 2)"""
        size = board.size
        if colour == Colour.RED:  # 连接上下 (Row 0 and Row size-1)
            return x <= 2 or x >= size - 3
        else:  # BLUE 连接左右 (Col 0 and Col size-1)
            return y <= 2 or y >= size - 3

    def _get_relevant_empty_tiles(self, board: Board) -> List[Tuple[int, int]]:
        """获取周围有棋子的空位 (用于 fallback 扫描)"""
        relevant = set()
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is not None:
                    # 将周围空位加入
                    for i in range(Tile.NEIGHBOUR_COUNT):
                        nx = x + Tile.I_DISPLACEMENTS[i]
                        ny = y + Tile.J_DISPLACEMENTS[i]
                        if 0 <= nx < size and 0 <= ny < size:
                            if board.tiles[nx][ny].colour is None:
                                relevant.add((nx, ny))
        return list(relevant)