import random

from board import NO_ONE

random.seed(42)


class RandomAgent:
    def __init__(self) -> None:
        self.player = NO_ONE

    @staticmethod
    def move(board):
        available_moves = board.available_moves()
        return random.choice(available_moves)
