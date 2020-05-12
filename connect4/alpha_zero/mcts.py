from copy import deepcopy
from math import sqrt
from random import choice
from board import C
from alpha_zero.util import encode_board
from board import PLAYER1, PLAYER2
import numpy as np

C_PUCT = 1
EPSILON = 0.25


class MCTS:  # noqa

    def __init__(self, net, player, turns) -> None:  # TODO megemelni a számott
        self.net = net
        self.player = player
        self.turns = turns

    def next_move(self, board, train=False):
        root = Node(board, self.player)
        e_board = root.compute(self.net, root=True, train=train)

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

        if train:
            return self._bets_action(root, train=train), (e_board, self._policy(root))
        else:
            return self._bets_action(root, train=train)

    def _search_best_leaf(self, node):
        if None in node.children.values() or len(node.children) == 0:
            return node
        return self._search_best_leaf(max(node.children.values(), key=lambda x: x.PUCT()))

    def _bets_action(self, node, train):
        if not train:
            key, _ = max(node.children.items(), key=lambda x: x[1].N)
        else:
            p = self._policy(node)  # TODO temperature
            key = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=p)
        return key

    @staticmethod
    def _policy(root):
        p = []
        children = root.children

        for i in range(C):
            p.append(children.get(i, DummyNode))

        return np.fromiter(map(lambda x: x.N / root.N, p), dtype=np.float32)


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

    def compute(self, net, root=False, train=False):
        e_board = encode_board(self.board)
        [p], [[v]] = net.model.predict_on_batch(np.expand_dims(e_board, axis=0))
        self.v = v
        if root:
            p = self._add_dirichlet_noise(p)
        for idx, value in enumerate(p):
            self.P[idx] = value
        if train:
            return e_board
        else:
            return self.v

    @staticmethod
    def _add_dirichlet_noise(p):
        return (1 - EPSILON) * p + \
               EPSILON * np.random.dirichlet([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])


class DummyNode:
    N = 0
