import multiprocessing as mp
import time

import numpy as np

from agent.alpha_z import AlphaZero
from alpha_zero.game_data import GameData
from board import PLAYER1, PLAYER2, Board, NO_ONE

THREAD_COUNT = 50


def predict_process(net, channels):
    print(f'Predict process started with net {net}')
    active_channels = channels
    n = len(active_channels)
    # Placed here, to not load in every process
    from alpha_zero.alpha_net import AlphaNet

    alpha_net = AlphaNet()  # TODO load if name

    while len(active_channels):
        boards = []
        t1 = time.perf_counter()
        buff_channels = active_channels.copy()
        for c in active_channels:
            data = c.recv()
            if isinstance(data, str) and data == 'Finished':
                print(f'Received player {c} finished')
                buff_channels.remove(c)
            else:
                boards.append(data)
        active_channels = buff_channels
        t2 = time.perf_counter()
        print(f'C: {round((t2 - t1) * 1000, 4)} ms')

        t3 = time.perf_counter()
        [policy, value] = alpha_net.predict(boards)
        policy, value = policy.numpy(), value.numpy()
        t4 = time.perf_counter()
        print(f'N: {round((t4 - t3) * 1000, 4)} ms')

        t5 = time.perf_counter()
        for idx, c in enumerate(active_channels):
            c.send([[policy[idx]], [value[idx]]])
        t6 = time.perf_counter()

        print(f'S: {round((t6 - t5) * 1000, 4)} ms')
        print(f'1: {round((t6 - t1) * 1000 / n, 4)} ms')


def multi_self_play(net, n=10, mcts_turns=100):
    pipes = [mp.Pipe() for _ in range(THREAD_COUNT)]

    predict_p = mp.Process(target=predict_process, args=(net, [pi[0] for pi in pipes]))
    predict_p.start()

    with mp.Pool(processes=THREAD_COUNT) as pool:
        data = pool.starmap(self_play, [(pi[1], n, mcts_turns, True) for pi in pipes])
        pool.close()
        pool.join()

    predict_p.kill()

    return data


def self_play(net, n=10, mcts_turns=100, multi_player=False):
    az_1 = AlphaZero(PLAYER1, net, mcts_turns=mcts_turns, multi_player=multi_player)
    az_2 = AlphaZero(PLAYER2, net, mcts_turns=mcts_turns, multi_player=multi_player)

    game_data = GameData()

    for i in range(n):
        print(f'self play {i + 1} starting')
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

    if multi_player:
        net.send('Finished')

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
    best_net = 'test_net'
    data = GameData('test_0')

    for i in range(times):
        # self play
        data_run = multi_self_play(net=best_net, n=10, mcts_turns=300)

        # add new data to database
        data.add_games(data_run)

        # save new database
        data.save(f'test_{i}')

        # retrain
        new_net = best_net.train(data, epochs=20)

        # evaluate
        best_net = evaluate(best_net, new_net)


def evaluate(net_1, net_2):
    return net_1  # TODO


if __name__ == '__main__':
    train(3)
