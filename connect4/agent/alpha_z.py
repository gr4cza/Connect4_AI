from alpha_zero.mcts import MCTS


class AlphaZero:

    def __init__(self, player, net=None, mcts_turns=100, multi_player=False):
        self.mcts = MCTS(net, player, turns=mcts_turns, multi_player=multi_player)

    def move(self, board, train=False):
        return self.mcts.next_move(board, train)
