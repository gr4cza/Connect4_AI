from __future__ import annotations

from copy import deepcopy
from math import sqrt
from random import choice

import numpy as np

from agent.random_agent import RandomAgent
from board import PLAYER1, PLAYER2, NO_ONE

CR = 1.4142  # ~ sqrt(2)
random_a = RandomAgent()


class Node(object):
    def __init__(self, board, player, parent=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.children = {}
        self.plays = 0
        self.wins = 0

        self._create_children_nodes()

    def _create_children_nodes(self):
        for move in self.board.available_moves():
            self.children[move] = None

    def UTC(self):
        utc = self.wins / self.plays + CR * sqrt(np.log(self.parent.plays) / self.plays)
        return utc

    def expand(self) -> Node:
        column = choice([key for (key, value) in self.children.items() if value is None])
        board = deepcopy(self.board)
        board.add_token(column)
        self.children[column] = Node(board, PLAYER1 if self.player == PLAYER2 else PLAYER2, self)
        return self.children[column]

    def simulate(self):
        board = deepcopy(self.board)
        while not board.is_game_over():
            col = random_a.move(board)
            board.add_token(col)
        return board.winner

    def back_propagate(self, winner):
        self.plays += 1
        self.wins += 1 if self.player != winner else 0
        if self.parent:
            self.parent.back_propagate(winner)

    def __str__(self) -> str:
        return f'{self.wins}/{self.plays} '


class MonteCarlo(object):
    def __init__(self, turns) -> None:
        self.turns = turns
        self.player = NO_ONE

    def move(self, board):
        return self._monte_carlo(board)

    def _monte_carlo(self, board):
        root = Node(board, self.player)
        for _ in range(self.turns):
            node = self._select_best_node(root)
            if not node.board.is_game_over():
                new_node = node.expand()
                winner = new_node.simulate()
                new_node.back_propagate(winner)
            else:
                node.back_propagate(node.board.winner)
        return self._best_move(root)

    @staticmethod
    def _best_move(root):
        key, _ = max(root.children.items(), key=lambda x: x[1].plays)
        return key

    def _select_best_node(self, node) -> Node:
        if None in node.children.values() or len(node.children) == 0:
            return node
        return self._select_best_node(max(node.children.values(), key=lambda x: x.UTC()))
