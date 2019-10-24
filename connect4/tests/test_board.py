import unittest

import numpy as np

from board import Board


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
        self.board.add_token(0)
        self.board.add_token(2)
        self.board.add_token(1)
        self.board.add_token(2)
        self.board.add_token(6)
        self.assertEqual(str(self.board), ('[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  ,  ,  ,  ,  ,  ]\n'
                                           '[ ,  , X,  ,  ,  ,  ]\n'
                                           '[O, O, X,  ,  ,  , O]\n'))

    def test_invalid_moves(self):
        self.board.board = np.array([[0, 0, 0, 2, 0, 0, 0],
                                     [0, 0, 0, 2, 0, 0, 0],
                                     [0, 0, 0, 2, 0, 0, 0],
                                     [0, 0, 0, 2, 0, 0, 0],
                                     [0, 0, 0, 2, 0, 0, 0],
                                     [0, 0, 0, 2, 0, 0, 0]])
        self.assertFalse(self.board.add_token(3))

    def test_empty_board_available_moves(self):
        self.assertEqual(self.board.available_moves(), [0, 1, 2, 3, 4, 5, 6])

    def test_one_full_column_available_moves(self):
        self.board.board = np.array([[0, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0]])
        self.assertEqual(self.board.available_moves(), [0, 2, 3, 4, 5, 6])

    def test_full_table_available_moves(self):
        self.board.board = np.array([[2, 1, 2, 1, 2, 1, 1],
                                     [2, 1, 2, 1, 2, 1, 1],
                                     [2, 1, 2, 1, 2, 1, 1],
                                     [2, 1, 2, 1, 2, 1, 1],
                                     [2, 1, 2, 1, 2, 1, 1],
                                     [2, 1, 2, 1, 2, 1, 1]])
        self.assertEqual(self.board.available_moves(), [])

    def test_no_winer_board(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 1],
                                     [1, 2, 1, 2, 2, 0, 2]])
        self.assertEqual(self.board.check_win(), 0)

    def test_player_one_win(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 0]])
        self.assertEqual(self.board.check_win(), 1)

    def test_player_two_win(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 0]])
        self.assertEqual(self.board.check_win(), 2)

    def test_row_win(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0],
                                     [0, 2, 1, 1, 1, 1, 2],
                                     [1, 2, 1, 2, 2, 1, 2]])
        self.assertEqual(self.board.check_win(), 1)

    def test_column_win(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0],
                                     [0, 2, 1, 1, 2, 1, 2],
                                     [1, 2, 1, 2, 2, 1, 2]])
        self.assertEqual(self.board.check_win(), 1)

    def test_last_column_win(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 2],
                                     [0, 0, 2, 0, 0, 0, 2],
                                     [0, 2, 1, 1, 2, 1, 2],
                                     [1, 2, 1, 2, 2, 1, 2]])
        self.assertEqual(self.board.check_win(), 2)

    def test_positive_diagonal(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 2, 0],
                                     [0, 2, 1, 1, 2, 1, 2],
                                     [1, 2, 1, 2, 2, 1, 2]])
        self.assertEqual(self.board.check_win(), 1)

    def test_negative_diagonal(self):
        self.board.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 2, 0, 1, 0],
                                     [0, 0, 0, 1, 2, 2, 0],
                                     [0, 2, 1, 1, 2, 2, 2],
                                     [1, 2, 1, 2, 2, 1, 2]])
        self.assertEqual(self.board.check_win(), 2)


if __name__ == '__main__':
    unittest.main()
