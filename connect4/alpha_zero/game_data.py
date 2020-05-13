import pickle

import numpy as np


class GameData:

    def __init__(self, file_name=None):
        self.boards = []
        self.policies = []
        self.values = []
        if file_name:
            self.load_from_file(file_name)

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

    def load_from_file(self, file_name):
        with open('./training_data/data/{}.pkl'.format(file_name), 'rb')as file:
            data = pickle.load(file)
        self.add_games([data])

    def save(self, file_name):
        with open('./training_data/data/{}.pkl'.format(file_name), 'wb')as file:
            pickle.dump(self, file)

    @property
    def board(self):
        return np.array(self.boards)

    @property
    def policy(self):
        return np.array(self.policies)

    @property
    def value(self):
        return np.array(self.values)

    def __str__(self) -> str:
        return 'Game_Data: {} states'.format(len(self.boards))


if __name__ == '__main__':
    gd = GameData('test')
    print(gd)
