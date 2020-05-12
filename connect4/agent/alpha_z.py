from alpha_zero.alpha_net import AlphaNet
from alpha_zero.mcts import MCTS


class AlphaZero:

    def __init__(self, player, net=None, mcts_turns=100):
        self.alpha_net = AlphaNet() if not net else net
        self.mcts = MCTS(self.alpha_net, player, turns=mcts_turns)

    def move(self, board, train=False):
        return self.mcts.next_move(board, train)

    def train(self):
        pass
