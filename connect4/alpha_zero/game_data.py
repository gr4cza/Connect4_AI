import json
import os
import pickle

import numpy as np
from pathlib import Path

BASE_DIR = f'{os.path.dirname(__file__)}/training_data/data/'


class GameData:

    def __init__(self, directory_name=None, max_games=50_000):
        self.boards = []
        self.policies = []
        self.values = []

        self.game_count = 0
        self.max_games = max_games
        self.game_lengths = []
        if directory_name:
            self.load_from_directory(directory_name)

    def add_play(self, board, policy):
        self.boards.append(board)
        self.policies.append(policy)

    def add_winner(self, value):
        length = len(self.boards)
        self.values.extend([value] * length)

        self.game_count += 1
        self.game_lengths.append(length)

    def add_games(self, games):
        for game in games:
            self.boards.extend(game.boards)
            self.policies.extend(game.policies)
            self.values.extend(game.values)
            self.game_lengths.extend(game.game_lengths)
            self.game_count += game.game_count

        self._purge()

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

    def _purge(self):
        if self.game_count > self.max_games:
            delta = self.game_count - self.max_games
            positions = sum(self.game_lengths[:delta])
            self.game_lengths = self.game_lengths[delta:]

            self.boards = self.boards[positions:]
            self.values = self.values[positions:]
            self.policies = self.policies[positions:]
            self.game_count = self.max_games

    def __str__(self) -> str:
        return f'Game_Data: {len(self.boards)} states in {self.game_count} games'

    def __len__(self):
        return len(self.values)


if __name__ == '__main__':
    gd = GameData('full_test_20200517_1857')
    print(gd)
