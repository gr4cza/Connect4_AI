import numpy as np
from gmpy2 import bit_set

NO_ONE = 0
PLAYER1 = 1
PLAYER2 = 2
R = 6
C = 7
RC = R * C


class Board:
    def __init__(self, board=None):
        if board is None:
            self.np_board = np.zeros([R, C], dtype=int)
        else:
            self.np_board = board
        self.bit_board_1 = 0
        self.bit_board_2 = 0

        self.row_counter = [R - 1 for _ in range(C)]
        self.bit_board_row_counter = [0, 7, 14, 21, 28, 35, 42]

        self.current_player = PLAYER1
        self.winner = NO_ONE
        self.move_count = 0

    def add_token(self, column):
        if self.winner == NO_ONE:
            if self.row_counter[column] < 0:
                print('Invalid move')
                return False

            player = self.current_player
            row, bit_row = self.__first_empty_row(column)
            self.np_board[row, column] = player
            if player == PLAYER1:
                self.bit_board_1 = bit_set(self.bit_board_1, bit_row)
            else:
                self.bit_board_2 = bit_set(self.bit_board_2, bit_row)

            self.move_count += 1

            self.check_winner()
            self.current_player = PLAYER1 if player != PLAYER1 else PLAYER2
            return True
        return False

    def is_game_over(self):
        return not self.winner == NO_ONE or not self.move_count <= RC - 1

    def available_moves(self):
        if self.winner == NO_ONE:
            return [i for i in range(C) if self.row_counter[i] >= 0]
        else:
            return []

    def __str__(self):
        table = ''
        for r in range(R):
            row = []
            for c in range(C):
                cel = self.np_board[r, c]
                if cel == NO_ONE:
                    row.append(' ')
                elif cel == PLAYER1:
                    row.append('O')
                elif cel == PLAYER2:
                    row.append('X')
            table += '[' + ', '.join(row) + ']\n'
        return table

    def __deepcopy__(self, memodict=None):
        copy_obj = Board(np.array(self.np_board))

        copy_obj.bit_board_1 = self.bit_board_1
        copy_obj.bit_board_2 = self.bit_board_2
        copy_obj.bit_board_row_counter = self.bit_board_row_counter.copy()

        copy_obj.current_player = self.current_player

        copy_obj.move_count = self.move_count
        copy_obj.row_counter = self.row_counter.copy()

        copy_obj.winner = self.winner
        return copy_obj

    def __first_empty_row(self, column):
        row = self.row_counter[column]
        self.row_counter[column] -= 1

        bit_row = self.bit_board_row_counter[column]
        self.bit_board_row_counter[column] += 1
        return row, bit_row

    def check_winner(self):
        bit_board = self.bit_board_1 if self.current_player == PLAYER1 else self.bit_board_2

        # check row
        m = bit_board & (bit_board >> 7)
        if m & (m >> 14):
            self.winner = self.current_player

        # check column
        m = bit_board & (bit_board >> 1)
        if m & (m >> 2):
            self.winner = self.current_player

        # check negative diagonal
        m = bit_board & (bit_board >> 6)
        if m & (m >> 12):
            self.winner = self.current_player

        # check positive diagonal
        m = bit_board & (bit_board >> 8)
        if m & (m >> 16):
            self.winner = self.current_player

    # for debug
    def get_hash(self):
        return np.array2string(self.np_board.flatten(), separator='')[1:-1]
