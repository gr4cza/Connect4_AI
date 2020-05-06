import numpy as np

from board import PLAYER1, PLAYER2


def encode_board(board):
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
    return encoded.reshape((1, 6, 7, 3))
