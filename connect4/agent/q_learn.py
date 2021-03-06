import math
import os
import pickle
import random
from copy import deepcopy
from datetime import datetime
from time import time

from connect4.board import Board, PLAYER2, PLAYER1, NO_ONE


class QLearn(object):
    def __init__(self, player, source_name=None):
        self.player = player
        self._states_value = self._load_learn_dict(source_name)

    def move(self, board):
        b_hash = board.get_hash()
        available = board.available_moves()
        scores = [(self._get_state_value(board.move_count, b_hash, col), col) for col in available]
        max_score = -math.inf
        if max(s[0] for s in scores) == 0:
            column = random.choice([s[1] for s in scores if s[0] == 0])
            self._get_state_value(board.move_count, b_hash, column)
        else:
            column = None
            for col in board.available_moves():
                score = self._get_state_value(board.move_count, b_hash, col)
                if score >= max_score:
                    max_score = score
                    column = col
        return column

    @staticmethod
    def _load_learn_dict(source_name):
        if source_name:
            with open(source_name, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def _get_state_value(self, moves, board, action):
        m = self._states_value.get(moves, None)
        if not m:
            return 0
        b = m.get(board, None)
        if not b:
            return 0
        c = b.get(action, None)
        if not c:
            return 0
        return c


class QLearnTrain(QLearn):
    def __init__(self, source_name=None):
        super().__init__(source_name)
        self.player = PLAYER1
        self._states = []
        self.learning_rate = 0.2
        self.exploration_rate = 0.3
        self.gamma_decay = 0.9
        self.win_loose = []
        self.resolution = 100

    def move(self, board):
        b_hash = board.get_hash()
        if random.uniform(0, 1) <= self.exploration_rate:
            column = random.choice(board.available_moves())
            self._get_state_value(board.move_count, b_hash, column)
        else:
            column = super().move(board)
        self._states.append((board.move_count, b_hash, column))
        return column

    def learn(self, iterations=100, against=None, name=''):
        current_player = PLAYER1
        self_play = False
        if not against:
            self_play = True
            against = QLearnTrain()
        against.player = PLAYER2

        self._iterate(against, current_player, iterations, self_play)
        if not self_play:
            self.player = PLAYER2
            against.player = PLAYER1
            temp = deepcopy(self._states_value)
            self._states_value = {}
            self._iterate(against, current_player, iterations, self_play)
            temp.update(self._states_value)
            self._states_value = temp

        if self_play:
            self._states_value.update(against._states_value)
            self.win_loose.append(against.win_loose)
        self._save_learn_dict(name)

    def _iterate(self, against, current_player, iterations, self_play):
        p1, p2, d = 0, 0, 0
        for i in range(iterations):
            if i % (iterations / 20) == 0:
                print(f'iteration: {i}.')

            if i % (iterations / self.resolution) == 0:
                self.win_loose.append((p1, p2, d, time()))

            winner = self._train(current_player, against)
            if self_play:
                against.__feed_reward(winner)
                against.__reset()
            current_player = PLAYER1

            if winner == 0:
                d += 1
            elif winner == PLAYER1:
                p1 += 1
            elif winner == PLAYER2:
                p2 += 1
        print(f'p1: {p1}, p2: {p2}, d: {d}')

    def _train(self, current_player, against):
        board = Board()
        while not board.is_game_over():
            if current_player == PLAYER1:
                if self.player == PLAYER1:
                    column = self.move(board)
                else:
                    column = against.move(board)
                current_player = PLAYER2
            else:
                if self.player == PLAYER2:
                    column = self.move(board)
                else:
                    column = against.move(board)
                current_player = PLAYER1
            board.add_token(column)
        winner = board.winner
        self.__feed_reward(winner)
        self.__reset()
        return winner

    def __feed_reward(self, winner):
        if winner == self.player:
            reward = 1.0
        elif winner == NO_ONE:
            reward = -0.1
        else:
            reward = -1.0
        for num_move, board, action in reversed(self._states):
            self._states_value[num_move][board][action] += self.learning_rate * (
                    self.gamma_decay * reward - self._states_value[num_move][board][action])
            reward = self._states_value[num_move][board][action]

    def _save_learn_dict(self, name):
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/learning'):
            os.makedirs('models/learning')
        file_name = f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(f'save model as: {file_name}')
        with open(f'models/{file_name}.pkl', 'wb') as f:
            pickle.dump(self._states_value, f)
        with open(f'models/learning/{file_name}_learning_rate.pkl', 'wb') as f:
            pickle.dump(self.win_loose, f)

    def _get_state_value(self, move_count, board, action):
        m = self._states_value.get(move_count, None)
        if not m:
            self._states_value[move_count] = {}
            m = self._states_value[move_count]
        b = m.get(board, None)
        if not b:
            self._states_value[move_count][board] = {}
            b = self._states_value[move_count][board]
        c = b.get(action, None)
        if not c:
            self._states_value[move_count][board][action] = 0
            return 0
        else:
            return c

    def __reset(self):
        self._states = []
        self._num_move = 0

    def reset(self):
        self._states_value = {}
        self.win_loose = []
