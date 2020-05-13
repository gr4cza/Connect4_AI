import numpy as np

from agent.alpha_z import AlphaZero
from alpha_zero.alpha_net import AlphaNet
from alpha_zero.game_data import GameData
from board import PLAYER1, PLAYER2, Board, NO_ONE


def self_play(net, n=10, mcts_turns=100):
    az_1 = AlphaZero(PLAYER1, net, mcts_turns=mcts_turns)
    az_2 = AlphaZero(PLAYER2, net, mcts_turns=mcts_turns)

    game_data = GameData()

    for i in range(n):
        print('self play {} starting'.format(i + 1))
        board = Board()
        current_player = PLAYER1

        p1_data = GameData()
        p2_data = GameData()

        while not board.is_game_over():
            if current_player == PLAYER1:
                action, (p1_board, p1_policy) = az_1.move(board, train=True)
                board.add_token(action)

                p1_data.add_play(p1_board, p1_policy)
                current_player = PLAYER2
            elif current_player == PLAYER2:
                action, (p2_board, p2_policy) = az_2.move(board, train=True)
                board.add_token(action)

                p2_data.add_play(p2_board, p2_policy)
                current_player = PLAYER1
        print(board)

        winner = board.winner

        p1_v = v_value(winner, PLAYER1)
        p1_data.add_winner(p1_v)

        p2_v = v_value(winner, PLAYER2)
        p2_data.add_winner(p2_v)

        game_data.add_games([p1_data, p2_data])

    return game_data


def v_value(winner, player):
    if winner == NO_ONE:
        return np.array([0], dtype=np.float32)
    if winner == player:
        return np.array([1], dtype=np.float32)
    if winner != player:
        return np.array([-1], dtype=np.float32)


def train(times):
    # variables
    best_net = AlphaNet()
    data = GameData()

    for _ in range(times):
        # self play
        data_run = self_play(net=best_net, n=10, mcts_turns=100)

        # add new data to database
        data.add_games([data_run])

        # retrain
        new_net = best_net.train(data, epochs=50)

        # evaluate
        best_net = evaluate(best_net, new_net)


def evaluate(net_1, net_2):
    return net_1  # TODO


if __name__ == '__main__':
    train(3)
