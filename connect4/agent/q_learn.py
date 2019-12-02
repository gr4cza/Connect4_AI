import math
import pickle
import random
from datetime import datetime

from board import Board, PLAYER2, PLAYER1, NO_ONE

random.seed(42)


class QLearn(object):
    def __init__(self, player=PLAYER1, source_name=None):
        self.player = player
        self._states_value = self._load_learn_dict(source_name)

    def move(self, board):
        b_hash = board.get_hash()
        available = board.available_moves()
        scores = [(self._get_state_value(board.get_moves(), b_hash, col), col) for col in available]
        if max(s[0] for s in scores) == 0:
            column = random.choice([s[1] for s in scores if s[0] == 0])
            self._get_state_value(board.get_moves(), b_hash, column)
        else:
            max_score = -math.inf
            column = None
            for col in board.available_moves():
                score = self._get_state_value(board.get_moves(), b_hash, col)
                if score >= max_score:
                    max_score = score
                    column = col
        return column, max_score

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
    def __init__(self, player=PLAYER1, source_name=None):
        super().__init__(player, source_name)
        self.states = []
        self.learning_rate = 0.2
        self.exploration_rate = 0.3
        self.gamma_decay = 0.9
        self._win_loose = []

    def move(self, board):
        b_hash = board.get_hash()
        if random.uniform(0, 1) <= self.exploration_rate:
            column = random.choice(board.available_moves())
            self._get_state_value(board.get_moves(), b_hash, column)
        else:
            column = super().move(board)
        self.states.append((board.get_moves(), b_hash, column))
        return column

    def learn(self, iterations=100, against=None, name=''):
        current_player = PLAYER1
        self_play = False
        if not against:
            self_play = True
            against = QLearnTrain(PLAYER2)

        p1, p2, d = 0, 0, 0

        for i in range(iterations):
            if i % (iterations / 20) == 0:
                print(f'iteration: {i}.')
            if i % (iterations / 1000) == 0:
                self._win_loose.append((p1, p2, d))

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

            winner = board.get_winner()

            self._feed_reward(winner)
            self._reset()
            if self_play:
                against._feed_reward(winner)
                against._reset()
            current_player = PLAYER1

            if winner == 0:
                d += 1
            elif winner == PLAYER1:
                p1 += 1
            elif winner == PLAYER2:
                p2 += 1

        print(f'p1: {p1}, p2: {p2}, d: {d}')

        self._save_learn_dict(name + ('_p1' if self_play else ''))
        if self_play:
            against._save_learn_dict(name + '_p2')

    def _feed_reward(self, winner):
        if winner == self.player:
            reward = 1.0
        elif winner == NO_ONE:
            reward = -0.1
        else:
            reward = -1.0
        for num_move, board, action in reversed(self.states):
            self._states_value[num_move][board][action] += self.learning_rate * (
                    self.gamma_decay * reward - self._states_value[num_move][board][action])  # TODO rework
            reward = self._states_value[num_move][board][action]

    def _save_learn_dict(self, name):
        file_name = f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(f'save model as: {file_name}')
        with open(f'models/{file_name}.pkl', 'wb') as f:
            pickle.dump(self._states_value, f)
        with open(f'models/learning/{file_name}_learning_rate.pkl', 'wb') as f:
            pickle.dump(self._win_loose, f)

    def _get_state_value(self, moves, board, action):
        m = self._states_value.get(moves, None)
        if not m:
            self._states_value[moves] = {}
            m = self._states_value[moves]
        b = m.get(board, None)
        if not b:
            self._states_value[moves][board] = {}
            b = self._states_value[moves][board]
        c = b.get(action, None)
        if not c:
            self._states_value[moves][board][action] = 0
            return 0
        else:
            return c

    def _reset(self):
        self.states = []
        self._num_move = 0

    def reset(self):
        self._states_value = {}
        self._win_loose = []
