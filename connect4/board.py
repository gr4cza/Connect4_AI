import numpy as np

NO_ONE = 0
PLAYER1 = 1
PLAYER2 = 2
R = 6
C = 7


class Board:
    def __init__(self, board=None):
        if board is None:
            self.board = np.zeros([R, C], dtype=int)
        else:
            self.board = board
        self.current_player = PLAYER1
        self.winner = NO_ONE
        self.move_count = 0
        self.row_counter = [R - 1 for _ in range(C)]
        self.moves = []  # for debug purposes

    def add_token(self, column):
        if self.winner == NO_ONE:
            if self.row_counter[column] < 0:
                print('Invalid move')
                return False
            row = self.__first_empty_row(column)
            self.board[row, column] = self.current_player
            self.current_player = PLAYER1 if self.current_player != PLAYER1 else PLAYER2
            self.__check_winner(row, column)
            self.move_count += 1
            self.moves.append(column)
            return True
        return False

    def is_game_over(self):
        return not self.winner == NO_ONE or not self.move_count <= R * C - 1

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
                cel = self.board[r, c]
                if cel == NO_ONE:
                    row.append(' ')
                elif cel == PLAYER1:
                    row.append('O')
                elif cel == PLAYER2:
                    row.append('X')
            table += '[' + ', '.join(row) + ']\n'
        return table

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        copy_obj = Board(np.array(self.board))
        copy_obj.current_player = self.current_player
        copy_obj.move_count = self.move_count
        copy_obj.row_counter = self.row_counter.copy()
        copy_obj.winner = self.winner
        return copy_obj

    def __first_empty_row(self, column):
        row = self.row_counter[column]
        self.row_counter[column] -= 1
        return row

    def __check_winner(self, row, column):
        # check row
        for c in range(column - 3 if column - 3 >= 0 else 0, column + 1 if column + 3 < 6 else 4):
            if self.board[row][c] == self.board[row][c + 1] == self.board[row][c + 2] == self.board[row][c + 3]:
                self.winner = self.board[row][column]

        # check column
        if row <= 2:
            if self.board[row][column] == self.board[row + 1][column] == \
                    self.board[row + 2][column] == self.board[row + 3][column]:
                self.winner = self.board[row][column]

        # check negative diagonal
        if 3 <= column + (5 - row) <= 8:
            for i in range(4):
                if 0 <= column - 3 + i and column + i <= 6 and 0 <= row - 3 + i and row + i <= 5:
                    if self.board[row + i][column + i] == self.board[row - 1 + i][column - 1 + i] == \
                            self.board[row - 2 + i][column - 2 + i] == self.board[row - 3 + i][column - 3 + i]:
                        self.winner = self.board[row][column]

        # check positive diagonal
        if 3 <= column + row <= 8:
            for i in range(4):
                if 0 <= column - 3 + i and column + i <= 6 and 0 <= row - i and row + 3 - i <= 5:
                    if self.board[row - i][column + i] == self.board[row + 1 - i][column - 1 + i] == \
                            self.board[row + 2 - i][column - 2 + i] == self.board[row + 3 - i][column - 3 + i]:
                        self.winner = self.board[row][column]

    # for debug
    def get_hash(self):
        return np.array2string(self.board.flatten(), separator='')[1:-1]
