import random
from src.Colour import Colour


class ZobristHash:
    def __init__(self, size: int, seed: int = 0):
        """
        Zobrist hashing with symmetry canonicalization.
        """
        self.size = size
        random.seed(seed)

        # table[x][y][v] where v:
        # 0 = empty (unused)
        # 1 = RED
        # 2 = BLUE
        self.table = [
            [
                [random.getrandbits(64) for _ in range(3)]
                for _ in range(size)
            ]
            for _ in range(size)
        ]

        # side to move
        self.side_table = {
            Colour.RED: random.getrandbits(64),
            Colour.BLUE: random.getrandbits(64),
        }

    # --------------------------------------------------
    # Board → immutable matrix
    # --------------------------------------------------
    def _board_to_matrix(self, board):
        mat = [[0] * self.size for _ in range(self.size)]
        for x in range(self.size):
            for y in range(self.size):
                c = board.tiles[x][y].colour
                if c is None:
                    mat[x][y] = 0
                elif c == Colour.RED:
                    mat[x][y] = 1
                else:
                    mat[x][y] = 2
        return tuple(tuple(row) for row in mat)

    # --------------------------------------------------
    # Symmetry transforms
    # --------------------------------------------------
    def _mirror_lr(self, mat):
        return tuple(
            tuple(mat[x][self.size - 1 - y] for y in range(self.size))
            for x in range(self.size)
        )

    def _mirror_ud(self, mat):
        return tuple(
            tuple(mat[self.size - 1 - x][y] for y in range(self.size))
            for x in range(self.size)
        )

    # --------------------------------------------------
    # Canonical Zobrist hash
    # --------------------------------------------------
    def get_hash(self, board, player: Colour) -> int:
        mat = self._board_to_matrix(board)

        variants = [
            mat,
            self._mirror_lr(mat),
            self._mirror_ud(mat),
            self._mirror_lr(self._mirror_ud(mat)),
        ]

        canonical = min(variants)

        h = 0
        for x in range(self.size):
            for y in range(self.size):
                v = canonical[x][y]
                if v != 0:
                    h ^= self.table[x][y][v]

        # side to move MUST be included
        h ^= self.side_table[player]
        return h
