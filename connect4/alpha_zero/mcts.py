from copy import deepcopy
from math import sqrt
from random import choice

from alpha_zero.util import encode_board
from board import PLAYER1, PLAYER2

C_PUCT = 1


class MCTS:

    def __init__(self, net, player, turns=100) -> None:  # TODO megemelni a számott
        self.net = net
        self.player = player
        self.turns = turns

    def next_move(self, board):
        root = Node(board, self.player)
        root.compute(self.net)

        for _ in range(self.turns):
            # select best node
            node = self._search_best_leaf(root)

            # if game over in leaf
            if node.board.is_game_over():
                node.back_propagate(node.v)
                continue

            # expand
            node = node.expand()
            v = node.compute(self.net)

            # back_propagate
            node.back_propagate(v)

        return self._bets_action(root)

    def _search_best_leaf(self, node):
        if None in node.children.values() or len(node.children) == 0:
            return node
        return self._search_best_leaf(max(node.children.values(), key=lambda x: x.PUCT()))

    @staticmethod
    def _bets_action(node):
        key, _ = max(node.children.items(), key=lambda x: x[1].N)
        return key


class Node:
    def __init__(self, board, player, action=-1, parent=None) -> None:
        self.board = board
        self.player = player
        self.action = action
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.P = {}
        self.v = 0.0

        self._create_children_nodes()

    def _create_children_nodes(self):
        for move in self.board.available_moves():
            self.children[move] = None

    def PUCT(self):  # noqa
        return (self.W / (1 + self.N)) + C_PUCT * self.parent.P[self.action] * \
               (sqrt(self.parent.N) / (1 + self.N))

    def expand(self):
        column = choice([key for (key, value) in self.children.items() if value is None])
        board = deepcopy(self.board)
        board.add_token(column)
        self.children[column] = Node(board, PLAYER1 if self.player == PLAYER2 else PLAYER2,
                                     action=column, parent=self)
        return self.children[column]

    def back_propagate(self, v):
        self.N += 1
        v1 = -v if self.board.current_player == PLAYER1 else v
        self.W += v1
        if self.parent:
            self.parent.back_propagate(v)

    def compute(self, net):
        [p], [[v]] = net.model.predict_on_batch(encode_board(self.board))
        self.v = v
        for idx, value in enumerate(p):
            self.P[idx] = value
        return self.v

    def __str__(self) -> str:
        return str(self.PUCT())
