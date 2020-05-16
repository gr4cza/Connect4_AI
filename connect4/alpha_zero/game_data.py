import json
import os
import pickle

import numpy as np
from pathlib import Path

BASE_DIR = f'{os.path.dirname(__file__)}/training_data/data/'


class GameData:

    def __init__(self, directory_name=None):
        self.boards = []
        self.policies = []
        self.values = []
        self.game_count = 0
        if directory_name:
            self.load_from_directory(directory_name)

    def add_play(self, board, policy):
        self.boards.append(board)
        self.policies.append(policy)

    def add_winner(self, value):
        self.values.extend([value] * len(self.boards))
        self.game_count += 1

    def add_games(self, games):
        for game in games:
            self.boards.extend(game.boards)
            self.policies.extend(game.policies)
            self.values.extend(game.values)
            self.game_count += game.game_count

    def load_from_directory(self, name):
        path = BASE_DIR + f'{name}/'

        last_iter = self._get_last_iteration(path)

        with open(path + f'{last_iter:02}.pkl', 'rb')as file:
            data = pickle.load(file)
        self.add_games([data])

    def save(self, name):
        path = BASE_DIR + f'{name}/'
        Path(path).mkdir(parents=True, exist_ok=True)

        last_iter = self._get_last_iteration(path, add=True)

        with open(path + f'{last_iter:02}.pkl', 'wb')as file:
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
        return f'Game_Data: {len(self.boards)} states in {self.game_count} games'

    @staticmethod
    def _get_last_iteration(path, add=False):
        if not os.path.exists(path + 'catalog.json'):
            with open(path + 'catalog.json', 'w')as f:
                data = {'last_item': 0}
                f.write(json.dumps(data))
                return data['last_item']
        else:
            with open(path + 'catalog.json', 'r+')as f:
                data = json.load(f)
                f.seek(0)
                if add:
                    data['last_item'] += 1
                f.write(json.dumps(data))
                return data['last_item']


if __name__ == '__main__':
    gd = GameData('test_20200516_0128')
    print(gd)
