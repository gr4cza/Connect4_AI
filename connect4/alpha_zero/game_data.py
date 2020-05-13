import numpy as np


class GameData:

    def __init__(self):
        self.boards = []
        self.policies = []
        self.values = []

    def add_play(self, board, policy):
        self.boards.append(board)
        self.policies.append(policy)

    def add_winner(self, value):
        self.values.extend([value] * len(self.boards))

    def add_games(self, games):
        for game in games:
            self.boards.extend(game.boards)
            self.policies.extend(game.policies)
            self.values.extend(game.values)

    @property
    def board(self):
        return np.array(self.boards)

    @property
    def policy(self):
        return np.array(self.policies)

    @property
    def value(self):
        return np.array(self.values)
