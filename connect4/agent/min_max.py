import math
import random
from copy import deepcopy

from board import NO_ONE

MAX_PLAYER = 1
MIN_PLAYER = -1
random.seed(42)


class MinMaxAgent(object):
    def __init__(self, depth, score_function):
        self.player = NO_ONE
        self.depth = depth
        self.score_function = score_function
        self.time_penalty = 0.99

    def move(self, board):
        score, move = self._min_max(board, self.depth)
        # print(f'ai choose score: {score:.4f}')
        return move

    def _min_max(self, board, depth, min_max_player=MAX_PLAYER):
        if depth == 0 or board.is_game_over():
            return self._score(board)

        if min_max_player == MAX_PLAYER:
            max_score = -math.inf
            column = None
            for col in board.available_moves():
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score, _ = self._min_max(board_copy, depth - 1, MIN_PLAYER)
                if max_score < score:
                    max_score = score
                    column = col
            return max_score, column

        if min_max_player == MIN_PLAYER:
            min_score = +math.inf
            column = None
            for col in board.available_moves():
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score, _ = self._min_max(board_copy, depth - 1, MAX_PLAYER)
                min_score = min(min_score, score)
                if min_score > score:
                    min_score = score
                    column = col
            return min_score, column

    def _score(self, board):
        if board.is_game_over():
            winner = board.winner
            if winner == self.player:
                score = 100
            elif winner == NO_ONE:
                score = 0
            else:
                score = -100
        else:
            score = self.score_function(board.board, self.player)
        return score, None


class MinMaxAgentWAlphaBeta(MinMaxAgent):
    def _min_max(self, board, depth, alpha=-math.inf, beta=math.inf, min_max_player=MAX_PLAYER):
        if depth == 0 or board.is_game_over():
            return self._score(board)

        if min_max_player == MAX_PLAYER:
            max_score = -math.inf
            column = None
            for col in board.available_moves():
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score, _ = self._min_max(board_copy, depth - 1, alpha, beta, MIN_PLAYER)
                score *= self.time_penalty
                if max_score < score:
                    max_score = score
                    column = col
                alpha = max(alpha, max_score)
                if alpha >= beta:
                    break
            return max_score, column

        if min_max_player == MIN_PLAYER:
            min_score = +math.inf
            column = None
            for col in board.available_moves():
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score, _ = self._min_max(board_copy, depth - 1, alpha, beta, MAX_PLAYER)
                score *= self.time_penalty
                if min_score > score:
                    min_score = score
                    column = col
                beta = min(beta, min_score)
                if alpha >= beta:
                    break
            return min_score, column
