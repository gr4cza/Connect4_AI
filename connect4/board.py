import numpy as np

PLAYER1 = 1
PLAYER2 = 2
R = 6
C = 7
WIN_LENGTH = 4


class Board:
    def __init__(self):
        self.board = np.zeros([R, C], dtype=int)
        self.current_player = PLAYER1

    def add_token(self, column):
        column = column
        if self.board[0, column] != 0:
            print('Invalid move')
            return False
        row = self._first_empty_row(column)
        self.board[row, column] = self.current_player
        self.current_player = PLAYER1 if self.current_player != PLAYER1 else PLAYER2
        return True

    def check_win(self):
        # check horizontal
        for r in range(R):
            for c in range(C - (WIN_LENGTH - 1)):
                if 0 != self.board[r][c] == self.board[r][c + 1] == self.board[r][c + 2] == self.board[r][c + 3]:
                    return self.board[r][c]

        # check vertical
        for c in range(C):
            for r in range(R - (WIN_LENGTH - 1)):
                if 0 != self.board[r][c] == self.board[r + 1][c] == self.board[r + 2][c] == self.board[r + 3][c]:
                    return self.board[r][c]

        # check diagonal
        for r in range(R - (WIN_LENGTH - 1)):
            for c in range(C - (WIN_LENGTH - 1)):
                if 0 != self.board[r][c] == self.board[r + 1][c + 1] == \
                        self.board[r + 2][c + 2] == self.board[r + 3][c + 3]:
                    return self.board[r][c]

                if 0 != self.board[r][c + 3] == self.board[r + 1][c + 2] == \
                        self.board[r + 2][c + 1] == self.board[r + 3][c]:
                    return self.board[r][c + 3]
        # if no win yet
        return 0

    def available_moves(self):
        return [i for i in range(C) if self.board[0, i] == 0]

    def __str__(self):
        table = ''
        for r in range(R):
            row = []
            for c in range(C):
                cel = self.board[r, c]
                if cel == 0:
                    row.append(' ')
                elif cel == 1:
                    row.append('O')
                elif cel == 2:
                    row.append('X')
            table += '[' + ', '.join(row) + ']\n'
        return table

    def _first_empty_row(self, column):
        row = 0
        while self.board[row, column] == 0:
            row += 1
            if row == 6:
                break
        row = row - 1
        return row
