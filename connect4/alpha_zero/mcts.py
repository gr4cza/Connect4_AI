from copy import deepcopy
from math import sqrt
from random import choice

import numpy as np
from scipy.special import softmax

from alpha_zero.util import encode_board
from board import C
from board import PLAYER1, PLAYER2

C_PUCT = 1
EPSILON = 0.25


class MCTS:  # noqa

    def __init__(self, net, player, turns, multi_process, print_policy=False, net_type=None):
        self.net = net
        self.player = player
        self.turns = turns
        self.multi_process = multi_process
        self.net_type = net_type
        self.print_policy = print_policy

    def next_move(self, board, train=False):
        if self.multi_process:
            root = MultiNode(board, self.player, root_player=self.player)
        elif self.net_type is not None:
            root = PlayAgainstNode(board, self.player,
                                   net_type=self.net_type, root_player=self.player)
        else:
            root = Node(board, self.player, root_player=self.player)

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
        if self.print_policy:
            print(self._policy(node))
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
    def __init__(self, board, player, root_player, action=-1, parent=None) -> None:
        self.board = board
        self.player = player
        self.action = action
        self.root_player = root_player
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
        self.children[column] = self.__class__(board, PLAYER1 if self.player == PLAYER2 else PLAYER2,
                                               action=column, parent=self, root_player=self.root_player)
        return self.children[column]

    def back_propagate(self, v):
        self.N += 1
        v1 = -v if self.player == self.root_player else v
        self.W += v1
        if self.parent:
            self.parent.back_propagate(v)

    def compute(self, net, root=False, train=False):
        e_board = encode_board(self.board)
        [p], [[v]] = self._predict(e_board, net)
        p = softmax(p)
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
    def _predict(e_board, net):
        return net.predict(e_board)

    @staticmethod
    def _add_dirichlet_noise(p):
        return (1 - EPSILON) * p + \
               EPSILON * np.random.dirichlet([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])


class MultiNode(Node):

    @staticmethod
    def _predict(e_board, net):
        net.send(e_board)
        rec = net.recv()
        return rec[0], rec[1]


class PlayAgainstNode(Node):

    def __init__(self, board, player, root_player, action=-1, parent=None, net_type=None) -> None:
        super().__init__(board, player, root_player, action, parent)
        self.net_type = net_type

    def _predict(self, e_board, net):
        data = {'type': self.net_type, 'board': e_board}
        net.send(data)
        rec = net.recv()
        return rec[0], rec[1]

    def expand(self):
        column = choice([key for (key, value) in self.children.items() if value is None])
        board = deepcopy(self.board)
        board.add_token(column)
        self.children[column] = self.__class__(board, PLAYER1 if self.player == PLAYER2 else PLAYER2,
                                               root_player=self.root_player,
                                               action=column,
                                               parent=self,
                                               net_type=self.net_type)
        return self.children[column]


class DummyNode:
    N = 0
