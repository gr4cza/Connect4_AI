import math
import random
from copy import deepcopy

from board import NO_ONE

MAX_PLAYER = 1
MIN_PLAYER = -1
random.seed(42)


class MinMaxAgent(object):
    def __init__(self, depth, calculate_score, player):
        self.player = player
        self.depth = depth
        self.calculate_score = calculate_score

    def move(self, board):
        score, move = self._min_max(board, self.depth, MAX_PLAYER)
        print(f'choose score{score}')
        return move

    def _min_max(self, board, depth, min_max_player):
        if depth == 0 or board.is_game_over():
            return self._score(board, min_max_player)

        if min_max_player == MAX_PLAYER:
            max_score = -math.inf
            available_moves = board.available_moves()
            column = random.choice(available_moves)
            for col in available_moves:
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score, _ = self._min_max(board_copy, depth - 1, MIN_PLAYER)
                if max_score < score:
                    max_score = score
                    column = col
            return max_score, column

        if min_max_player == MIN_PLAYER:
            min_score = +math.inf
            available_moves = board.available_moves()
            column = random.choice(available_moves)
            for col in available_moves:
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score, _ = self._min_max(board_copy, depth - 1, MAX_PLAYER)
                min_score = min(min_score, score)
                if min_score > score:
                    min_score = score
                    column = col
            return min_score, column

    def _score(self, board, min_max_player):
        if board.is_game_over():
            winner = board.get_winner()
            if winner == self.player:
                return 10_000, None
            elif winner == NO_ONE:
                return 0, None
            else:
                return -10_000, None
        else:
            return self.calculate_score(board.board, self.player), None


class MinMaxAgentWAlphaBeta(MinMaxAgent):
    def move(self, board):
        score, move = self._min_max_alpha_beta(board, self.depth, -math.inf, math.inf, MAX_PLAYER)
        print(f'choose score {score}')
        return move

    def _min_max_alpha_beta(self, board, depth, alpha, beta, min_max_player):
        if depth == 0 or board.is_game_over():
            return self._score(board, min_max_player)

        if min_max_player == MAX_PLAYER:
            max_score = -math.inf
            available_moves = board.available_moves()
            column = random.choice(available_moves)
            for col in available_moves:
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score = self._min_max_alpha_beta(board_copy, depth - 1, alpha, beta, MIN_PLAYER)[0]
                if max_score < score:
                    max_score = score
                    column = col
                alpha = max(alpha, max_score)
                if alpha >= beta:
                    break
            return max_score, column

        if min_max_player == MIN_PLAYER:
            min_score = +math.inf
            available_moves = board.available_moves()
            column = random.choice(available_moves)
            for col in available_moves:
                board_copy = deepcopy(board)
                board_copy.add_token(col)
                score = self._min_max_alpha_beta(board_copy, depth - 1, alpha, beta, MAX_PLAYER)[0]
                if min_score > score:
                    min_score = score
                    column = col
                alpha = min(alpha, min_score)
                if alpha >= beta:
                    break
            return min_score, column
