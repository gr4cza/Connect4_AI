import random


class RandomAgent:
    def __init__(self) -> None:
        super().__init__()
        self.player = None

    @staticmethod
    def move(board):
        available_moves = board.available_moves()
        return random.choice(available_moves)
