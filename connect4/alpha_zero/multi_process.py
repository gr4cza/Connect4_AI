import datetime
import json
import math
import multiprocessing as mp
import os
import random
import time
import gc

import numpy as np

from agent.alpha_z import AlphaZero
from alpha_zero.game_data import GameData
from board import PLAYER1, PLAYER2, Board, NO_ONE

THREAD_COUNT = 5

BASE_DIR = f'{os.path.dirname(__file__)}/training_data/models/'


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

        predict_and_send(alpha_net, boards, active_channels)
        gc.collect()


def match_process(net_name, channels):
    print(f'Match process started with "{net_name}"')
    active_channels = channels

    # Placed here, to not load in every process
    from alpha_zero.alpha_net import AlphaNet
    alpha_net_latest = AlphaNet(net_name, is_latest=True)
    alpha_net_best = AlphaNet(net_name, is_latest=False)

    while len(active_channels) != 0:
        boards_latest = []
        boards_best = []

        channel_latest = []
        channel_best = []

        buff_channels = active_channels.copy()
        for c in active_channels:
            data = c.recv()
            if isinstance(data, str) and data == 'Finished':
                print(f'Received player {c} finished')
                buff_channels.remove(c)
            else:
                net_type = data.get('type')
                if net_type == 'latest':
                    channel_latest.append(c)
                    boards_latest.append(data['board'])
                elif net_type == 'best':
                    channel_best.append(c)
                    boards_best.append(data['board'])
        active_channels = buff_channels

        # latest
        predict_and_send(alpha_net_latest, boards_latest, channel_latest)

        # best
        predict_and_send(alpha_net_best, boards_best, channel_best)
        gc.collect()


def predict_and_send(net, boards, channels):
    [policy, value] = net.predict(boards)
    policy, value = policy.numpy(), value.numpy()

    for c, p, v in zip(channels, policy, value):
        c.send([[p], [v]])


def multi_self_play(net_name, hours, minutes, mcts_turns):
    pipes = [mp.Pipe() for _ in range(THREAD_COUNT)]

    predict_p = mp.Process(target=predict_process, args=(net_name, [pi[0] for pi in pipes]))
    predict_p.start()

    with mp.Pool(processes=THREAD_COUNT) as pool:
        data = pool.starmap(self_play, [(pi[1], hours, minutes, mcts_turns, True) for pi in pipes])
        pool.close()
        pool.join()

    predict_p.join()

    return data


def evaluate(net_name, times, mcts_turns):
    pipes = [mp.Pipe() for _ in range(THREAD_COUNT)]

    match_p = mp.Process(target=match_process, args=(net_name, [pi[0] for pi in pipes]))
    match_p.start()

    times_per_process = math.ceil(times / THREAD_COUNT)

    with mp.Pool(processes=THREAD_COUNT)as pool:
        data = pool.starmap(play_against,
                            [(pi[1], times_per_process, mcts_turns, idx > (THREAD_COUNT / 2))
                             for idx, pi in enumerate(pipes)])
        pool.close()
        pool.join()

    l = sum([d[0] for d in data])
    b = sum([d[1] for d in data])
    t = sum([d[2] for d in data])
    print(f'Latest:{l} Best:{b} Tie:{t}')

    match_p.join()

    if l / (l + b) > 0.55:
        update_best_net_to_latest(net_name)


def train_net_process(net_name, epochs):
    data = GameData(net_name)

    from alpha_zero.alpha_net import AlphaNet
    loaded_net = AlphaNet(net_name)
    # retrain
    loaded_net.train(data, epochs=epochs)

    # close net
    loaded_net.release()


def self_play(net, hours, minutes, mcts_turns, multi_process=False):
    az_1 = AlphaZero(PLAYER1, net, mcts_turns=mcts_turns, multi_process=multi_process)
    az_2 = AlphaZero(PLAYER2, net, mcts_turns=mcts_turns, multi_process=multi_process)

    game_data = GameData()
    pid = os.getpid()

    # seed randoms with pid+time
    seed = int(pid + time.time())
    random.seed(seed)
    np.random.seed(seed)

    end_time = time.time() + datetime.timedelta(hours=hours,
                                                minutes=minutes).seconds

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
        gc.collect()

    if multi_process:
        net.send('Finished')

    return game_data


def play_against(net, times, mcts_turns, best_first):
    player_1 = AlphaZero(PLAYER1, net,
                         mcts_turns=mcts_turns,
                         net_type='latest')
    player_2 = AlphaZero(PLAYER2, net,
                         mcts_turns=mcts_turns,
                         net_type='best')

    if best_first:
        player_1, player_2 = player_2, player_1

    p1, p2, t = 0, 0, 0

    pid = os.getpid()

    # seed randoms with pid+time
    seed = int(pid + time.time())
    random.seed(seed)
    np.random.seed(seed)

    for i in range(times):
        print(f'[{pid}] play against {i + 1} starting')
        board = Board()
        current_player = PLAYER1

        while not board.is_game_over():
            if current_player == PLAYER1:
                action = player_1.move(board)
                board.add_token(action)
                current_player = PLAYER2
            elif current_player == PLAYER2:
                action = player_2.move(board)
                board.add_token(action)
                current_player = PLAYER1
        print(f'O:{"l" if not best_first else "b"} X:{"b" if not best_first else "l"}')
        print(board)

        winner = board.winner
        if winner == PLAYER1:
            p1 += 1
        elif winner == PLAYER2:
            p2 += 1
        elif winner == NO_ONE:
            t += 1

    net.send('Finished')

    if not best_first:
        return p1, p2, t
    else:
        return p2, p1, t


def v_value(winner, player):
    if winner == NO_ONE:
        return np.array([0], dtype=np.float32)
    if winner == player:
        return np.array([1], dtype=np.float32)
    if winner != player:
        return np.array([-1], dtype=np.float32)


def update_best_net_to_latest(net_name):
    with open(f'{BASE_DIR}{net_name}/catalog.json', 'r+')as f:
        data = json.load(f)
        f.seek(0)
        data['best_net'] = data['latest_net']
        f.write(json.dumps(data))
