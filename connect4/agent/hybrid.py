import random

from agent.min_max import MinMaxAgentWAlphaBeta
from agent.q_learn import QLearn
from board import NO_ONE

MAX_PLAYER = 1
MIN_PLAYER = -1
random.seed(42)


class HybridAgent(object):
    def __init__(self, depth, score_function, source_name):
        self.player = NO_ONE
        self.min_max = MinMaxAgentWAlphaBeta(depth, score_function)
        self.q_learn = QLearn(source_name=source_name)

    def move(self, board):
        move = self._hybrid(board)
        # print(f'ai choose score: {score:.4f}')
        return move

    def _hybrid(self, board, ):
        column, score = self.q_learn.move(board)
        if score == 0:
            column = self.min_max.move(board)
        return column
