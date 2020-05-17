from alpha_zero.mcts import MCTS


class AlphaZero:

    def __init__(self, player, net,
                 mcts_turns=100,
                 multi_process=False,
                 print_policy=False,
                 net_type=None):
        self.mcts = MCTS(net, player,
                         turns=mcts_turns,
                         multi_process=multi_process,
                         print_policy=print_policy,
                         net_type=net_type)

    def move(self, board, train=False):
        return self.mcts.next_move(board, train)
