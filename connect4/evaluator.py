from itertools import product

from numpy import count_nonzero

from board import NO_ONE, PLAYER1, PLAYER2, R, C


class BasicScore:
    @staticmethod
    def score(*_):
        return 0


class AdvancedScore:
    def __init__(self):
        self.point_dict = self.generate_points()

    def score(self, board, player, middle=True):
        sum_score = 0
        parity = 1
        if player != PLAYER1:
            parity = -1

        # check horizontal
        for r in range(R):
            if not all(x == board[r][0] for x in board[r][1:]):
                for c in range(C - 4):
                    sum_score += self.point_dict[
                                     (board[r][c], board[r][c + 1], board[r][c + 2], board[r][c + 3])] * parity

        # check vertical
        for c in range(C):
            for r in range(R - 4):
                sum_score += self.point_dict[(board[r][c], board[r + 1][c], board[r + 2][c], board[r + 3][c])] * parity

        # check diagonal
        for r in range(R - 4):
            for c in range(C - 4):
                sum_score += self.point_dict[
                                 (board[r][c], board[r + 1][c + 1], board[r + 2][c + 2], board[r + 3][c + 3])] * parity
                sum_score += self.point_dict[
                                 (board[r][c - 3], board[r + 1][c - 2], board[r + 2][c - 1], board[r + 3][c])] * parity

        if middle:
            sum_score += count_nonzero(board[-1][3] == player)
        return sum_score

    @staticmethod
    def generate_points():
        point_dict = {}
        combinations = [com for com in product([NO_ONE, PLAYER1, PLAYER2], repeat=4)]
        for com in combinations:
            p1_count = com.count(PLAYER1)
            p2_count = com.count(PLAYER2)
            if p1_count == 2 and p2_count == 0:
                point_dict[com] = 3
            elif p2_count == 2 and p1_count == 0:
                point_dict[com] = -3
            elif p1_count == 3 and p2_count == 0:
                point_dict[com] = 12
            elif p2_count == 3 and p1_count == 0:
                point_dict[com] = -12
            else:
                point_dict[com] = 0

        return point_dict
