from alpha_zero.mcts import MCTS


class AlphaZero:

    def __init__(self, player, net, mcts_turns=100, multi_player=False, print_policy=False):
        self.mcts = MCTS(net, player, turns=mcts_turns, multi_player=multi_player, print_policy=print_policy)

    def move(self, board, train=False):
        return self.mcts.next_move(board, train)
