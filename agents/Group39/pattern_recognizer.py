"""
Pattern Recognizer (upgraded, still lightweight)
- True hex bridge / cut points
- parallel/double bridge
- center bias (smooth decay)
- feature-based prior via distance maps (0-1 BFS)
- weak VC (component-to-component min-cost == 2)
- weak blocking (not overlapping ThreatDetector)
"""
from __future__ import annotations
from collections import deque
from typing import Dict, List, Set, Tuple, Optional

from agents.Group39.phase import GamePhase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile


INF = 10**9


class PatternRecognizer:
    def __init__(self, colour: Colour):
        self.colour = colour
        self._cache_key = None
        self._cache = None

    # =========================================================
    # Cache
    # =========================================================
    def _get_cache(self, board: Board, colour: Colour):
        from .utils import hash_board
        key = (hash_board(board), colour)
        if key == self._cache_key and self._cache is not None:
            return self._cache

        # 1) bridges (true hex bridge) + parallel counts
        bridge_points, bridge_count = self._bridge_points_and_counts(board, colour)

        # 2) cut points = opponent bridge points
        opp = Colour.opposite(colour)
        cut_points, cut_count = self._bridge_points_and_counts(board, opp)

        # 3) connection points near goal edges (keep your old one, but cheap)
        conn_points = set(self.detect_connection_patterns(board, colour))

        # 4) distance maps for feature-based prior
        my_s, my_t, my_best = self._distance_maps(board, colour)
        opp_s, opp_t, opp_best = self._distance_maps(board, opp)

        # 5) weak VC points (min cost == 2 between components)
        weak_vc = self._weak_vc_points(board, colour, max_components=6)

        self._cache_key = key
        self._cache = {
            "bridge_points": bridge_points,          # Set[(x,y)]
            "bridge_count": bridge_count,            # Dict[(x,y)] -> int
            "cut_points": cut_points,                # Set[(x,y)]
            "cut_count": cut_count,                  # Dict[(x,y)] -> int
            "conn_points": conn_points,              # Set[(x,y)]
            "weak_vc": weak_vc,                      # Set[(x,y)]
            "my_s": my_s, "my_t": my_t, "my_best": my_best,
            "opp_s": opp_s, "opp_t": opp_t, "opp_best": opp_best,
        }
        return self._cache

    # =========================================================
    # Core patterns
    # =========================================================
    def _bridge_points_and_counts(self, board: Board, colour: Colour) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """
        True Hex bridge:
        Two friendly stones at (x,y) and (ex,ey) such that they are '2-step apart' with two intermediate empty bridge legs.
        We detect by: pick a stone, pick two different neighbor directions d1,d2,
        if end cell (x+d1+d2) is friendly => bridge legs are (x+d1) and (x+d2).
        """
        pts: Set[Tuple[int, int]] = set()
        cnt: Dict[Tuple[int, int], int] = {}

        dirs = [(Tile.I_DISPLACEMENTS[i], Tile.J_DISPLACEMENTS[i]) for i in range(Tile.NEIGHBOUR_COUNT)]

        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour != colour:
                    continue

                # choose two directions
                for i in range(len(dirs)):
                    dx1, dy1 = dirs[i]
                    a1x, a1y = x + dx1, y + dy1
                    if not (0 <= a1x < board.size and 0 <= a1y < board.size):
                        continue

                    for j in range(i + 1, len(dirs)):
                        dx2, dy2 = dirs[j]
                        a2x, a2y = x + dx2, y + dy2
                        if not (0 <= a2x < board.size and 0 <= a2y < board.size):
                            continue

                        ex, ey = x + dx1 + dx2, y + dy1 + dy2
                        if not (0 <= ex < board.size and 0 <= ey < board.size):
                            continue

                        if board.tiles[ex][ey].colour != colour:
                            continue

                        # bridge legs must be empty (or at least not opponent)
                        legs = [(a1x, a1y), (a2x, a2y)]
                        for lx, ly in legs:
                            if board.tiles[lx][ly].colour is None:
                                pts.add((lx, ly))
                                cnt[(lx, ly)] = cnt.get((lx, ly), 0) + 1

        return pts, cnt

    def detect_cut_bridge_points(self, board: Board, opponent_colour: Colour):
        pts, _ = self._bridge_points_and_counts(board, opponent_colour)
        return list(pts)

    # =========================================================
    # Connection points (keep cheap)
    # =========================================================
    def detect_connection_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        connection_points = []
        if colour == Colour.RED:
            # top & bottom bands
            rows = [0, 1, 2, board.size - 3, board.size - 2, board.size - 1]
            for x in rows:
                if 0 <= x < board.size:
                    for y in range(board.size):
                        if board.tiles[x][y].colour is None and self._is_near_my_pieces(board, x, y, colour, 2):
                            connection_points.append((x, y))
        else:
            cols = [0, 1, 2, board.size - 3, board.size - 2, board.size - 1]
            for y in cols:
                if 0 <= y < board.size:
                    for x in range(board.size):
                        if board.tiles[x][y].colour is None and self._is_near_my_pieces(board, x, y, colour, 2):
                            connection_points.append((x, y))
        return connection_points

    # =========================================================
    # Weak VC (component-to-component min cost == 2)
    # =========================================================
    def _weak_vc_points(self, board: Board, colour: Colour, max_components: int = 6) -> Set[Tuple[int, int]]:
        """
        Weak VC heuristic:
        - find own stone components (connected groups)
        - pick up to K largest components
        - for each pair, run 0-1 BFS between components:
            own stones cost 0, empty cost 1, opponent blocked
          if min_cost == 2, reconstruct one shortest path and collect empty cells on it.
        """
        comps = self._get_components(board, colour)
        if len(comps) < 2:
            return set()

        comps.sort(key=lambda c: len(c), reverse=True)
        comps = comps[:max_components]

        weak_pts: Set[Tuple[int, int]] = set()

        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                path_empties = self._min_cost_path_empties_between_components(board, colour, comps[i], comps[j], target_cost=2)
                if path_empties:
                    for p in path_empties:
                        weak_pts.add(p)

        return weak_pts

    def _get_components(self, board: Board, colour: Colour) -> List[Set[Tuple[int, int]]]:
        vis = set()
        comps = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour != colour or (x, y) in vis:
                    continue
                q = deque([(x, y)])
                vis.add((x, y))
                comp = {(x, y)}
                while q:
                    cx, cy = q.popleft()
                    for k in range(Tile.NEIGHBOUR_COUNT):
                        nx, ny = cx + Tile.I_DISPLACEMENTS[k], cy + Tile.J_DISPLACEMENTS[k]
                        if 0 <= nx < board.size and 0 <= ny < board.size:
                            if (nx, ny) not in vis and board.tiles[nx][ny].colour == colour:
                                vis.add((nx, ny))
                                comp.add((nx, ny))
                                q.append((nx, ny))
                comps.append(comp)
        return comps

    def _min_cost_path_empties_between_components(
        self,
        board: Board,
        colour: Colour,
        compA: Set[Tuple[int, int]],
        compB: Set[Tuple[int, int]],
        target_cost: int = 2
    ) -> Optional[List[Tuple[int, int]]]:
        """
        0-1 BFS from compA to reach any node in compB.
        cost: own=0, empty=1, opp=INF
        reconstruct one shortest path when reaching compB.
        """
        opp = Colour.opposite(colour)
        dist = [[INF] * board.size for _ in range(board.size)]
        parent = [[None] * board.size for _ in range(board.size)]
        dq = deque()

        for (sx, sy) in compA:
            dist[sx][sy] = 0
            dq.appendleft((sx, sy))

        target_set = set(compB)

        def node_cost(x: int, y: int) -> int:
            c = board.tiles[x][y].colour
            if c == opp:
                return INF
            if c == colour:
                return 0
            return 1

        reached = None

        while dq:
            x, y = dq.popleft()
            if dist[x][y] > target_cost:
                continue
            if (x, y) in target_set:
                reached = (x, y)
                break

            for k in range(Tile.NEIGHBOUR_COUNT):
                nx, ny = x + Tile.I_DISPLACEMENTS[k], y + Tile.J_DISPLACEMENTS[k]
                if not (0 <= nx < board.size and 0 <= ny < board.size):
                    continue

                cst = node_cost(nx, ny)
                if cst >= INF:
                    continue

                nd = dist[x][y] + cst
                if nd < dist[nx][ny] and nd <= target_cost:
                    dist[nx][ny] = nd
                    parent[nx][ny] = (x, y)
                    if cst == 0:
                        dq.appendleft((nx, ny))
                    else:
                        dq.append((nx, ny))

        if reached is None:
            return None
        if dist[reached[0]][reached[1]] != target_cost:
            return None

        # reconstruct empties along path
        empties = []
        cx, cy = reached
        while parent[cx][cy] is not None:
            if board.tiles[cx][cy].colour is None:
                empties.append((cx, cy))
            cx, cy = parent[cx][cy]
        empties.reverse()

        # we only want exactly 2 empties for "weak VC" signal
        if len(empties) == 2:
            return empties
        return None

    # =========================================================
    # Feature-based maps
    # =========================================================
    def _distance_maps(self, board: Board, colour: Colour):
        """
        Compute 0-1 BFS dist from start edge and from target edge.
        Return: dist_from_start, dist_to_target, best_cost.
        """
        opp = Colour.opposite(colour)

        def cell_cost(x: int, y: int) -> int:
            c = board.tiles[x][y].colour
            if c == opp:
                return INF
            if c == colour:
                return 0
            return 1

        # start sources / target predicate
        if colour == Colour.RED:
            start_nodes = [(0, y) for y in range(board.size)]
            target_nodes = [(board.size - 1, y) for y in range(board.size)]
        else:
            start_nodes = [(x, 0) for x in range(board.size)]
            target_nodes = [(x, board.size - 1) for x in range(board.size)]

        distS = self._zero_one_bfs(board, start_nodes, cell_cost)
        distT = self._zero_one_bfs(board, target_nodes, cell_cost)

        best = INF
        for x in range(board.size):
            for y in range(board.size):
                if distS[x][y] < INF and distT[x][y] < INF:
                    # if this cell is empty, stepping onto it costs 1; if occupied by us costs 0 already included
                    best = min(best, distS[x][y] + distT[x][y] - (1 if board.tiles[x][y].colour is None else 0))
        return distS, distT, best

    def _zero_one_bfs(self, board: Board, sources: List[Tuple[int, int]], cost_fn):
        dist = [[INF] * board.size for _ in range(board.size)]
        dq = deque()

        for sx, sy in sources:
            cst = cost_fn(sx, sy)
            if cst >= INF:
                continue
            dist[sx][sy] = cst
            if cst == 0:
                dq.appendleft((sx, sy))
            else:
                dq.append((sx, sy))

        while dq:
            x, y = dq.popleft()
            for k in range(Tile.NEIGHBOUR_COUNT):
                nx, ny = x + Tile.I_DISPLACEMENTS[k], y + Tile.J_DISPLACEMENTS[k]
                if not (0 <= nx < board.size and 0 <= ny < board.size):
                    continue
                cst = cost_fn(nx, ny)
                if cst >= INF:
                    continue
                nd = dist[x][y] + cst
                if nd < dist[nx][ny]:
                    dist[nx][ny] = nd
                    if cst == 0:
                        dq.appendleft((nx, ny))
                    else:
                        dq.append((nx, ny))
        return dist

    # =========================================================
    # Public API: detect_simple_patterns (fallback use)
    # =========================================================
    def detect_simple_patterns(self, board: Board, colour: Colour) -> List[Tuple[int, int, float]]:
        cache = self._get_cache(board, colour)
        out = []
        for (x, y) in cache["bridge_points"]:
            out.append((x, y, 0.85))
        for (x, y) in cache["weak_vc"]:
            out.append((x, y, 0.75))
        for (x, y) in cache["conn_points"]:
            out.append((x, y, 0.60))
        for (x, y) in cache["cut_points"]:
            out.append((x, y, 0.70))
        return out

    # =========================================================
    # Prior (the important part)
    # =========================================================
    def get_prior(self, board: Board, move: Move, colour: Colour, phase: GamePhase) -> float:
        if not (0 <= move.x < board.size and 0 <= move.y < board.size):
            return 0.0
        if board.tiles[move.x][move.y].colour is not None:
            return 0.0

        cache = self._get_cache(board, colour)
        x, y = move.x, move.y

        prior = 0.03

        # ---------- 1) center bias (smooth) ----------
        center = board.size // 2
        d = abs(x - center) + abs(y - center)
        if d <= 1:
            prior += 0.28
        elif d <= 2:
            prior += 0.18
        elif d <= 3:
            prior += 0.10
        elif d <= 4:
            prior += 0.05

        # opening slightly boosts center; late slightly reduces
        if phase == GamePhase.OPENING:
            prior += 0.05
        elif phase == GamePhase.LATEGAME:
            prior -= 0.03

        # ---------- 2) bridge / parallel bridge ----------
        if (x, y) in cache["bridge_points"]:
            # base bridge bonus
            if phase == GamePhase.OPENING:
                prior += 0.14
            elif phase == GamePhase.MIDGAME:
                prior += 0.24
            else:
                prior += 0.30

            # parallel/double bridge: count>=2 means this point supports multiple bridges
            bc = cache["bridge_count"].get((x, y), 0)
            if bc >= 2:
                prior += 0.10
            if bc >= 3:
                prior += 0.06

        # ---------- 3) weak VC points ----------
        if (x, y) in cache["weak_vc"]:
            prior += 0.16 if phase != GamePhase.LATEGAME else 0.20

        # ---------- 4) connection points (edge progress) ----------
        if (x, y) in cache["conn_points"]:
            prior += 0.10 if phase == GamePhase.OPENING else 0.08

        # ---------- 5) weak blocking (cut points) ----------
        # 注意：这里只是“静态弱阻断”，不要做 ThreatDetector 的强制阻断
        if (x, y) in cache["cut_points"]:
            prior += 0.10
            cc = cache["cut_count"].get((x, y), 0)
            if cc >= 2:
                prior += 0.05

        # ---------- 6) feature-based: path corridor / improvement ----------
        my_s, my_t, my_best = cache["my_s"], cache["my_t"], cache["my_best"]
        opp_s, opp_t, opp_best = cache["opp_s"], cache["opp_t"], cache["opp_best"]

        # my improvement: if placing here likely reduces my best path cost
        if my_s[x][y] < INF and my_t[x][y] < INF and my_best < INF:
            # putting a stone makes this cell cost 0 instead of 1
            est_new = my_s[x][y] + my_t[x][y] - 1
            gain = max(0, my_best - est_new)
            if gain > 0:
                prior += min(0.12, 0.06 * gain)  # cap

        # opp damage: if this cell is in opponent corridor (close to their best)
        if opp_s[x][y] < INF and opp_t[x][y] < INF and opp_best < INF:
            corridor_score = (opp_s[x][y] + opp_t[x][y] - 1) - opp_best
            # corridor_score small => closer to best path corridor
            if corridor_score <= 1:
                prior += 0.10
            elif corridor_score <= 2:
                prior += 0.06

        # close to opponent key group (local tactical pressure)
        opp = Colour.opposite(colour)
        adj_opp = 0
        for k in range(Tile.NEIGHBOUR_COUNT):
            nx, ny = x + Tile.I_DISPLACEMENTS[k], y + Tile.J_DISPLACEMENTS[k]
            if 0 <= nx < board.size and 0 <= ny < board.size:
                if board.tiles[nx][ny].colour == opp:
                    adj_opp += 1
        if adj_opp == 1:
            prior += 0.04
        elif adj_opp >= 2:
            prior += 0.07

        # phase safety caps (avoid over-bias in opening)
        if phase == GamePhase.OPENING:
            prior = min(prior, 0.75)
        return max(0.0, min(1.0, prior))

    # =========================================================
    # Helpers
    # =========================================================
    def _is_near_my_pieces(self, board: Board, x: int, y: int, colour: Colour, distance: int = 2) -> bool:
        if distance <= 0:
            return False
        q = deque([(x, y, 0)])
        vis = {(x, y)}
        while q:
            cx, cy, d = q.popleft()
            if d >= distance:
                continue
            for k in range(Tile.NEIGHBOUR_COUNT):
                nx, ny = cx + Tile.I_DISPLACEMENTS[k], cy + Tile.J_DISPLACEMENTS[k]
                if 0 <= nx < board.size and 0 <= ny < board.size and (nx, ny) not in vis:
                    vis.add((nx, ny))
                    if board.tiles[nx][ny].colour == colour:
                        return True
                    if board.tiles[nx][ny].colour is None:
                        q.append((nx, ny, d + 1))
        return False
