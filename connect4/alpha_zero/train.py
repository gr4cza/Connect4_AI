import numpy as np

from agent.alpha_z import AlphaZero
from alpha_zero.alpha_net import AlphaNet
from board import PLAYER1, PLAYER2, Board, NO_ONE


def self_play(n=10, net=None, mcts_train=100):
    net = net if net else AlphaNet()
    az_1 = AlphaZero(PLAYER1, net, mcts_turns=mcts_train)
    az_2 = AlphaZero(PLAYER2, net, mcts_turns=mcts_train)

    game_data = ([], [], [])

    for i in range(n):
        print('self play {} starting'.format(i + 1))
        board = Board()
        current_player = PLAYER1

        p1_data = ([], [], [])
        p2_data = ([], [], [])

        while not board.is_game_over():
            if current_player == PLAYER1:
                action, (p1_board, p1_policy) = az_1.move(board, train=True)
                board.add_token(action)

                p1_data[0].append(p1_board)
                p1_data[1].append(p1_policy)
                current_player = PLAYER2
            elif current_player == PLAYER2:
                action, (p2_board, p2_policy) = az_2.move(board, train=True)
                board.add_token(action)

                p2_data[0].append(p2_board)
                p2_data[1].append(p2_policy)
                current_player = PLAYER1
            print(board)

        winner = board.winner
        print(winner)
        p1_v = v_value(winner, PLAYER1)
        p1_data[2].extend([p1_v] * len(p1_data[0]))

        p2_v = v_value(winner, PLAYER2)
        p2_data[2].extend([p2_v] * len(p2_data[0]))

        for idx, t in enumerate(game_data):
            t += p1_data[idx]
            t += p2_data[idx]

    return game_data


def v_value(winner, player):
    if winner == NO_ONE:
        return np.array([0], dtype=np.float32)
    if winner == player:
        return np.array([1], dtype=np.float32)
    if winner != player:
        return np.array([-1], dtype=np.float32)


if __name__ == '__main__':
    alpha_net = AlphaNet()
    data_run = self_play(10, alpha_net, mcts_train=100)
    board = np.array(data_run[0])
    policy = np.array(data_run[1])
    value = np.array(data_run[2])
    alpha_net.model.fit(board, {'policy_out': policy, 'value_out': value}, epochs=3)

    data_run = self_play(10, alpha_net, mcts_train=100)

    # size = 1000
    # random_data = np.random.randint(2, size=(size, 6, 7, 3)).astype(np.float32)
    # ran_p = softmax(np.random.uniform(low=-1.0, high=1.0, size=(size, 7)), axis=1)
    # ran_v = np.random.randint(low=-1, high=1, size=size).astype(np.float32)
    # h = alpha_net.model.fit(random_data, [ran_p, ran_v], batch_size=5, epochs=3, verbose=2)
    # print(h)
