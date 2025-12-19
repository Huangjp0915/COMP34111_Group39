from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour


class HumanAgent(AgentBase):
    """
    Console human player.
    Input formats:
      - "x y"   (e.g. 5 5)
      - "x,y"   (e.g. 5,5)
      - "swap"  (only meaningful on turn 2 for the second player)
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        size = board.size

        while True:
            s = input(f"[YOU {self.colour}] Enter move (x y / x,y / swap): ").strip().lower()

            if s in {"swap", "s"}:
                # In this framework, swap is represented as Move(-1, -1)
                # Engine will reject it if it's illegal, but we still gate it for sanity:
                if turn == 2 and (not opp_move or not opp_move.is_swap()):
                    return Move(-1, -1)
                print("Swap is not legal now (usually only turn 2 for player2). Try again.")
                continue

            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) != 2:
                print("Bad input. Use: x y  (e.g. 5 5) or 'swap'")
                continue

            try:
                x = int(parts[0])
                y = int(parts[1])
            except ValueError:
                print("Bad input. x and y must be integers.")
                continue

            if not (0 <= x < size and 0 <= y < size):
                print(f"Out of bounds. x,y must be in [0, {size-1}].")
                continue

            if board.tiles[x][y].colour is not None:
                print("That cell is occupied. Try again.")
                continue

            return Move(x, y)