import datetime
import multiprocessing as mp
import os
import random
import time

import numpy as np

from agent.alpha_z import AlphaZero
from alpha_zero.game_data import GameData
from board import PLAYER1, PLAYER2, Board, NO_ONE

THREAD_COUNT = 5


def predict_process(net_name, channels):
    print(f'Predict process started with "{net_name}"')
    active_channels = channels

    # Placed here, to not load in every process
    from alpha_zero.alpha_net import AlphaNet
    alpha_net = AlphaNet(net_name)

    while len(active_channels) != 0:
        boards = []

        buff_channels = active_channels.copy()
        for c in active_channels:
            data = c.recv()
            if isinstance(data, str) and data == 'Finished':
                print(f'Received player {c} finished')
                buff_channels.remove(c)
            else:
                boards.append(data)
        active_channels = buff_channels

        [policy, value] = alpha_net.predict(boards)
        policy, value = policy.numpy(), value.numpy()

        for c, p, v in zip(active_channels, policy, value):
            c.send([[p], [v]])


def multi_self_play(net_name, hours=10, mcts_turns=100):
    pipes = [mp.Pipe() for _ in range(THREAD_COUNT)]

    predict_p = mp.Process(target=predict_process, args=(net_name, [pi[0] for pi in pipes]))
    predict_p.start()

    with mp.Pool(processes=THREAD_COUNT) as pool:
        data = pool.starmap(self_play, [(pi[1], hours, mcts_turns, True) for pi in pipes])
        pool.close()
        pool.join()

    predict_p.kill()

    return data


def self_play(net, hours=10, mcts_turns=100, multi_player=False):
    az_1 = AlphaZero(PLAYER1, net, mcts_turns=mcts_turns, multi_player=multi_player)
    az_2 = AlphaZero(PLAYER2, net, mcts_turns=mcts_turns, multi_player=multi_player)

    game_data = GameData()
    pid = os.getpid()

    # seed randoms with pid+time
    random.seed(pid+time.time())
    np.random.seed(pid+time.time())

    end_time = time.time() + datetime.timedelta(hours=hours).seconds

    i = 0
    while time.time() < end_time:
        i += 1
        print(f'[{pid}] self play {i} starting')
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


def evaluate(net_1, net_2=None):
    return False  # TODO
