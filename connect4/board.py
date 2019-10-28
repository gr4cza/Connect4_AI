import numpy as np

NO_ONE = 0
PLAYER1 = 1
PLAYER2 = 2
R = 6
C = 7
WIN_LENGTH = 4


class Board:
    def __init__(self):
        self.board = np.zeros([R, C], dtype=int)
        self.current_player = PLAYER1
        self._winner = NO_ONE

    def add_token(self, column):
        column = column
        if self.board[0, column] != 0:
            print('Invalid move')
            return False
        row = self._first_empty_row(column)
        self.board[row, column] = self.current_player
        self.current_player = PLAYER1 if self.current_player != PLAYER1 else PLAYER2
        self._check_winner(row, column)
        return True

    def check_win(self):
        return self._winner

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

    def _check_winner(self, row, column):
        # check row
        for c in range(column - 3 if column - 3 >= 0 else 0, column if column + 3 <= 6 else 3):
            if self.board[row][c] == self.board[row][c + 1] == self.board[row][c + 2] == self.board[row][c + 3]:
                self._winner = self.board[row][column]
                print('eq row')

        # check column
        if row <= 2:
            if self.board[row][column] == self.board[row + 1][column] == \
                    self.board[row + 2][column] == self.board[row + 3][column]:
                self._winner = self.board[row][column]
                print('eq col')

        # check negative diagonal
        if 3 <= column + (5 - row) <= 8:
            for i in range(4):
                if 0 <= column - 3 + i and column + i <= 6 and 0 <= row - 3 + i and row + i <= 5:
                    if self.board[row + i][column + i] == self.board[row - 1 + i][column - 1 + i] == \
                            self.board[row - 2 + i][column - 2 + i] == self.board[row - 3 + i][column - 3 + i]:
                        print('eq neg')
                        self._winner = self.board[row][column]

        # check positive diagonal
        if 3 <= column + row <= 8:
            for i in range(4):
                if 0 <= column - 3 + i and column + i <= 6 and 0 <= row - i and row + 3 - i <= 5:
                    if self.board[row - i][column + i] == self.board[row + 1 - i][column - 1 + i] == \
                            self.board[row + 2 - i][column - 2 + i] == self.board[row + 3 - i][column - 3 + i]:
                        print('eq pos')
                        self._winner = self.board[row][column]
