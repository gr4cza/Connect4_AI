import random


class RandomAgent:
    @staticmethod
    def move(board):
        available_moves = board.available_moves()
        return random.choice(available_moves)
