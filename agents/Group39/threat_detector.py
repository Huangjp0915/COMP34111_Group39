# threat_detector.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile

from .connectivity_evaluator import ConnectivityEvaluator


class ThreatDetector:
    """
    战术层 ThreatDetector（可落地版本）
    - 1-ply 全盘必胜/必防
    - 桥的战术响应
    - 2-ply forcing / double threat
    - cut threat（基于最短路代价变化的近似）
    """

    def __init__(self, colour: Colour):
        self.colour = colour
        self.ev = ConnectivityEvaluator()

    # ----------------------------
    # Low-level safe simulation
    # ----------------------------
    def _try_move_has_ended(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        if board.tiles[x][y].colour is not None:
            return False
        original = board.tiles[x][y].colour
        board.tiles[x][y].colour = colour
        # 清理 winner cache（你原本就这么做）
        board._winner = None
        ok = board.has_ended(colour)
        board.tiles[x][y].colour = original
        board._winner = None
        return ok

    def _iter_empty(self, board: Board):
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    yield x, y

    # ----------------------------
    # 1) Improved 1-ply win/lose
    # ----------------------------
    def immediate_win_moves(self, board: Board, colour: Colour, limit: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        全盘扫描：所有一手必胜点
        limit: 用于 rollout/模拟时做快速版本（例如 limit=20）
        """
        wins = []
        checked = 0
        for x, y in self._iter_empty(board):
            if self._try_move_has_ended(board, x, y, colour):
                wins.append((x, y))
            checked += 1
            if limit is not None and checked >= limit:
                break
        return wins

    def detect_immediate_threats(self, board: Board, opponent_colour: Colour) -> List[Tuple[int, int, str]]:
        """
        兼容旧接口：返回 [(x,y,"WIN"), (x,y,"LOSE"), ...]
        WIN：我方下一手直接赢
        LOSE：对手下一手直接赢（我方必须下这里挡）
        """
        threats: List[Tuple[int, int, str]] = []

        my_wins = self.immediate_win_moves(board, self.colour, limit=None)
        if my_wins:
            # 立即赢：返回一个就够了（SmartHexAgent 会直接下）
            x, y = my_wins[0]
            return [(x, y, "WIN")]

        opp_wins = self.immediate_win_moves(board, opponent_colour, limit=None)
        for x, y in opp_wins:
            threats.append((x, y, "LOSE"))
        return threats

    # ----------------------------
    # 4) Bridge tactical logic
    # ----------------------------
    def _bridge_points_for_pair(self, a: Tuple[int,int], b: Tuple[int,int]) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
        """
        给定两枚同色棋子坐标，若它们形成“经典桥形”，返回两枚桥点（两个空位）。
        实现方式：找它们的共同邻居（hex 上最多 2 个共同邻居）。
        """
        ax, ay = a
        bx, by = b

        neigh_a = set()
        for i in range(Tile.NEIGHBOUR_COUNT):
            neigh_a.add((ax + Tile.I_DISPLACEMENTS[i], ay + Tile.J_DISPLACEMENTS[i]))

        common = []
        for i in range(Tile.NEIGHBOUR_COUNT):
            nx, ny = bx + Tile.I_DISPLACEMENTS[i], by + Tile.J_DISPLACEMENTS[i]
            if (nx, ny) in neigh_a:
                common.append((nx, ny))

        # 经典桥通常有两个共同邻接点
        if len(common) != 2:
            return None
        return common[0], common[1]

    def bridge_responses(self, board: Board, colour: Colour) -> List[Tuple[int,int]]:
        """
        若对手点了我方桥点的一侧，我方应立刻补另一侧，保持“虚连”成立。
        返回所有必须补的桥响应点（通常很少）。
        """
        my_stones = [(x, y) for x in range(board.size) for y in range(board.size) if board.tiles[x][y].colour == colour]
        if len(my_stones) < 2:
            return []

        opp = Colour.opposite(colour)
        responses: Set[Tuple[int,int]] = set()

        # 粗暴 O(n^2) 扫同色棋子对，11x11 数量不大可接受
        for i in range(len(my_stones)):
            for j in range(i+1, len(my_stones)):
                bp = self._bridge_points_for_pair(my_stones[i], my_stones[j])
                if not bp:
                    continue
                (p1x,p1y), (p2x,p2y) = bp
                # 越界跳过
                if not (0 <= p1x < board.size and 0 <= p1y < board.size and 0 <= p2x < board.size and 0 <= p2y < board.size):
                    continue

                c1 = board.tiles[p1x][p1y].colour
                c2 = board.tiles[p2x][p2y].colour

                # 对手占一个桥点，另一个空 => 立刻补
                if c1 == opp and c2 is None:
                    responses.add((p2x,p2y))
                if c2 == opp and c1 is None:
                    responses.add((p1x,p1y))

        return list(responses)

    # ----------------------------
    # 3) True double threat / forcing (2-ply)
    # ----------------------------
    def forcing_win_moves_2ply(self, board: Board, colour: Colour, candidates: Optional[List[Tuple[int,int]]] = None,
                              max_candidates: int = 18) -> List[Tuple[int,int]]:
        """
        找 2-ply forcing：
        我下 A -> 对手必须挡（否则我下一手赢） -> 即使对手挡了，我仍存在一手必胜点

        candidates：可传入候选空点列表（比如来自 pattern/prior 的前 K），否则默认取中心附近+邻近己子（简单降支）。
        """
        opp = Colour.opposite(colour)

        if candidates is None:
            # 简单候选生成：中心附近 + 邻近己方棋
            c = board.size // 2
            cand_set: Set[Tuple[int,int]] = set()
            for x, y in self._iter_empty(board):
                if abs(x - c) + abs(y - c) <= 3:
                    cand_set.add((x,y))
            # 邻近己子
            for sx in range(board.size):
                for sy in range(board.size):
                    if board.tiles[sx][sy].colour == colour:
                        for k in range(Tile.NEIGHBOUR_COUNT):
                            nx, ny = sx + Tile.I_DISPLACEMENTS[k], sy + Tile.J_DISPLACEMENTS[k]
                            if 0 <= nx < board.size and 0 <= ny < board.size and board.tiles[nx][ny].colour is None:
                                cand_set.add((nx,ny))
            candidates = list(cand_set)

        # 截断候选
        candidates = candidates[:max_candidates]

        forcing = []

        for ax, ay in candidates:
            if board.tiles[ax][ay].colour is not None:
                continue

            # 下 A
            board.tiles[ax][ay].colour = colour
            board._winner = None

            # 我方下一手有哪些必胜点？
            my_wins = self.immediate_win_moves(board, colour, limit=None)

            # 1) “强 double threat”：必胜点 >= 2（对手一手挡不完）
            if len(my_wins) >= 2:
                forcing.append((ax, ay))
                board.tiles[ax][ay].colour = None
                board._winner = None
                continue

            # 2) “必须应对”且应对后仍赢：若 my_wins == 1，对手必须下在那个点
            if len(my_wins) == 1:
                bx, by = my_wins[0]

                # 对手挡 B
                board.tiles[bx][by].colour = opp
                board._winner = None

                # 我方是否还有一手必胜点？（如果有，则 A 是 forcing）
                my_wins_after_block = self.immediate_win_moves(board, colour, limit=None)
                if my_wins_after_block:
                    forcing.append((ax, ay))

                # undo block
                board.tiles[bx][by].colour = None
                board._winner = None

            # undo A
            board.tiles[ax][ay].colour = None
            board._winner = None

        return forcing

    # ----------------------------
    # 2) Cut threat (practical approximation)
    # ----------------------------
    def cut_moves(self, board: Board, my_colour: Colour, top_k: int = 10) -> List[Tuple[int,int]]:
        """
        近似 cut：
        找那些“我占据后显著增加对手 shortest_path_cost”的点。
        为了不全盘爆算，可以只在“对手最短路带”上做（这里实现先全盘，再按增量排序，11x11仍可接受）。
        """
        opp = Colour.opposite(my_colour)
        before = self.ev.shortest_path_cost(board, opp)
        if before == float("inf"):
            return []

        scored: List[Tuple[float, Tuple[int,int]]] = []

        for x, y in self._iter_empty(board):
            # 我方占据该点
            board.tiles[x][y].colour = my_colour
            board._winner = None
            after = self.ev.shortest_path_cost(board, opp)
            board.tiles[x][y].colour = None
            board._winner = None

            if after == float("inf"):
                delta = 999.0
            else:
                delta = after - before

            # “切断/显著增代价”才算
            if delta >= 2.0:
                scored.append((delta, (x,y)))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [xy for _, xy in scored[:top_k]]
