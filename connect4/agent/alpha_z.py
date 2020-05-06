from alpha_zero.alpha_net import AlphaNet
from alpha_zero.mcts import MCTS
from board import Board


class AlphaZero:

    def __init__(self, player):
        self.player = player
        self.alpha_net = AlphaNet()
        self.mcts = MCTS(self.alpha_net, player)

    def move(self, board):
        mv = self._predict(board)
        return mv

    def train(self):
        pass

    def _predict(self, board):
        return self.mcts.next_move(board)


if __name__ == '__main__':
    a = AlphaZero()
    b = Board()
    print(a._encode_board(b))