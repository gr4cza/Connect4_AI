import math

from agent.min_max import MinMaxAgentWAlphaBeta
from agent.q_learn import QLearn
from board import NO_ONE

MAX_PLAYER = 1
MIN_PLAYER = -1


class HybridAgent(object):
    def __init__(self, depth, score_function, source_name):
        self.player = NO_ONE
        self.min_max = MinMaxAgentWAlphaBeta(depth, score_function)
        self.q_learn = QLearn(source_name=source_name)

    def move(self, board):
        move = self._hybrid(board)
        return move

    def _hybrid(self, board, ):
        column, score = self.q_learn.move(board)
        if score == -math.inf:
            column = self.min_max.move(board)
        return column


#TODO remove