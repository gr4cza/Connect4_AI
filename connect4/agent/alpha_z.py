import numpy as np

from alpha_zero.alpha_net import AlphaNet
from board import PLAYER1, PLAYER2, Board


class AlphaZero:

    def __init__(self):
        self.alpha_net = AlphaNet()

    def move(self, board):
        mv = self._predict(board)
        return mv

    def train(self):
        pass

    def _predict(self, board):
        encoded = self._encode_board(board)
        p, _ = self.alpha_net.model.predict(encoded.reshape((1, 6, 7, 3)))
        return np.argmax(p)

    @staticmethod
    def _encode_board(board):
        b = board.np_board
        encoded = np.zeros([6, 7, 3])
        for row in range(6):
            for col in range(7):
                if b[row, col] == PLAYER1:
                    encoded[row, col, 0] = 1
                elif b[row, col] == PLAYER2:
                    encoded[row, col, 1] = 1
        if board.current_player == PLAYER2:
            encoded[:, :, 2] = 1
        return encoded


if __name__ == '__main__':
    a = AlphaZero()
    b = Board()
    print(a._encode_board(b))