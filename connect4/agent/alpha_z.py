from alpha_zero.alpha_net import AlphaNet
from alpha_zero.mcts import MCTS
from board import Board


class AlphaZero:

    def __init__(self, player, net=None):
        self.alpha_net = AlphaNet() if not net else net
        self.mcts = MCTS(self.alpha_net, player)

    def move(self, board):
        return self.mcts.next_move(board)

    def train(self):
        pass
