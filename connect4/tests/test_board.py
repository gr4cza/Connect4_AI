import unittest
from copy import deepcopy

import numpy as np

from board import Board, NO_ONE, PLAYER1, PLAYER2


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.board = Board()

    def test_empty_board(self):
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'))

    def test_first_move(self):
        self.board.add_token(0)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[O,  ,  ,  ,  ,  ,  ]\n'))

    def test_multiple_moves(self):
        for i in [0, 2, 1, 2, 6]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  , X,  ,  ,  ,  ]\n'
                                           '[O, O, X,  ,  ,  , O]\n'))

    def test_invalid_moves(self):
        for i in [1, 1, 1, 1, 1, 1]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ , X,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'
                                           '[ , X,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'
                                           '[ , X,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'))
        self.assertFalse(self.board.add_token(1))

    def test_empty_board_available_moves(self):
        self.assertEqual(self.board.available_moves(), [0, 1, 2, 3, 4, 5, 6])

    def test_one_full_column_available_moves(self):
        for i in [1, 1, 1, 1, 1, 1]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ , X,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'
                                           '[ , X,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'
                                           '[ , X,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'))
        self.assertEqual(self.board.available_moves(), [0, 2, 3, 4, 5, 6])

    def test_full_table_available_moves(self):
        for i in [3, 3, 3, 4, 1, 0, 0, 3, 2, 0, 6, 1, 4, 2, 1, 4, 4, 1, 0, 5, 5, 3, 2, 0, 3, 2, 1,
                  4, 2, 5, 4, 0, 5, 5, 1, 6, 5, 2, 6, 6, 6, 6]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[X, O, X, O, O, O, X]\n'
                                           '[X, O, O, X, X, X, O]\n'
                                           '[O, X, X, X, O, O, X]\n'
                                           '[X, O, O, O, X, X, O]\n'
                                           '[O, X, X, X, O, O, X]\n'
                                           '[X, O, O, O, X, X, O]\n'))
        self.assertEqual(self.board.available_moves(), [])

    def test_after_win_available_moves(self):
        for i in [2, 1, 2, 1, 2, 1, 2]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  , O,  ,  ,  ,  ]\n'
                                           '[ , X, O,  ,  ,  ,  ]\n'
                                           '[ , X, O,  ,  ,  ,  ]\n'
                                           '[ , X, O,  ,  ,  ,  ]\n'))
        self.assertEqual(self.board.is_game_over(), True)
        self.assertFalse(self.board.add_token(3))
        self.assertEqual(self.board.available_moves(), [])

    def test_no_winner_board(self):
        for i in [0, 1, 2, 3, 2, 3, 2, 4, 6, 6]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  , O,  ,  ,  ,  ]\n'
                                           '[ ,  , O, X,  ,  , X]\n'
                                           '[O, X, O, X, X,  , O]\n'))
        self.assertEqual(self.board.winner, NO_ONE)
        self.assertEqual(self.board.is_game_over(), False)

    def test_player_one_win(self):
        for i in [1, 6, 2, 6, 3, 5, 4]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  , X]\n'
                                           '[ , O, O, O, O, X, X]\n'))
        self.assertEqual(self.board.winner, PLAYER1)
        self.assertEqual(self.board.is_game_over(), True)

    def test_player_two_win(self):
        for i in [0, 1, 6, 2, 6, 3, 5, 4]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  , O]\n'
                                           '[O, X, X, X, X, O, O]\n'))
        self.assertEqual(self.board.winner, PLAYER2)
        self.assertEqual(self.board.is_game_over(), True)

    def test_row_win(self):
        for i in [0, 1, 2, 1, 2, 3, 4, 5, 6, 1, 3, 6, 5, 0, 4]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ , X,  ,  ,  ,  ,  ]\n'
                                           '[X, X, O, O, O, O, X]\n'
                                           '[O, X, O, X, O, X, O]\n'))
        self.assertEqual(self.board.winner, PLAYER1)

    def test_column_win(self):
        for i in [0, 1, 2, 1, 2, 3, 2, 4, 2]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  , O,  ,  ,  ,  ]\n'
                                           '[ ,  , O,  ,  ,  ,  ]\n'
                                           '[ , X, O,  ,  ,  ,  ]\n'
                                           '[O, X, O, X, X,  ,  ]\n'))
        self.assertEqual(self.board.winner, PLAYER1)

    def test_column_not_bottom_win(self):
        for i in [0, 1, 1, 2, 1, 2, 1, 2, 1]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'
                                           '[ , O,  ,  ,  ,  ,  ]\n'
                                           '[ , O, X,  ,  ,  ,  ]\n'
                                           '[ , O, X,  ,  ,  ,  ]\n'
                                           '[O, X, X,  ,  ,  ,  ]\n'))
        self.assertEqual(self.board.winner, PLAYER1)

    def test_positive_diagonal(self):
        for i in [0, 1, 1, 2, 2, 3, 2, 3, 3, 5, 3]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  , O,  ,  ,  ]\n'
                                           '[ ,  , O, O,  ,  ,  ]\n'
                                           '[ , O, O, X,  ,  ,  ]\n'
                                           '[O, X, X, X,  , X,  ]\n'))
        self.assertEqual(self.board.winner, PLAYER1)

    def test_negative_diagonal(self):
        for i in [0, 0, 0, 0, 1, 1, 2, 1, 4, 2, 6, 3]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[X,  ,  ,  ,  ,  ,  ]\n'
                                           '[O, X,  ,  ,  ,  ,  ]\n'
                                           '[X, X, X,  ,  ,  ,  ]\n'
                                           '[O, O, O, X, O,  , O]\n'))
        self.assertEqual(self.board.winner, PLAYER2)

    def test_diagonal_center(self):
        for i in [3, 2, 4, 1, 1, 2, 2, 3, 3, 4, 3, 4, 6, 4, 4]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  , O,  ,  ]\n'
                                           '[ ,  ,  , O, X,  ,  ]\n'
                                           '[ ,  , O, O, X,  ,  ]\n'
                                           '[ , O, X, X, X,  ,  ]\n'
                                           '[ , X, X, O, O,  , O]\n'))
        self.assertEqual(self.board.winner, PLAYER1)

    def test_draw(self):
        for i in [3, 3, 3, 4, 1, 0, 0, 3, 2, 0, 6, 1, 4, 2, 1, 4, 4, 1, 0, 5, 5, 3, 2, 0, 3, 2, 1,
                  4, 2, 5, 4, 0, 5, 5, 1, 6, 5, 2, 6, 6, 6, 6]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[X, O, X, O, O, O, X]\n'
                                           '[X, O, O, X, X, X, O]\n'
                                           '[O, X, X, X, O, O, X]\n'
                                           '[X, O, O, O, X, X, O]\n'
                                           '[O, X, X, X, O, O, X]\n'
                                           '[X, O, O, O, X, X, O]\n'))
        self.assertEqual(self.board.winner, NO_ONE)

    def test_board_hash(self):
        for i in [3, 2, 4, 1, 1, 2, 2, 3, 3]:
            self.board.add_token(i)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  , O, O,  ,  ,  ]\n'
                                           '[ , O, X, X,  ,  ,  ]\n'
                                           '[ , X, X, O, O,  ,  ]\n'))
        self.assertEqual(self.board.get_hash(), '000000000000000000000001100001220000221100')

    def test_deep_copy(self):
        for i in [3, 2, 4, 1, 1, 2, 2, 3, 3, 5, 4, 6]:
            self.board.add_token(i)
        copy = deepcopy(self.board)
        self.assertNotEqual(self.board, copy)
        self.assertEqual(str(self.board), str(copy))


if __name__ == '__main__':
    unittest.main()
