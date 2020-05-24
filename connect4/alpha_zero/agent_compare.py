import math
import multiprocessing as mp
import os
import random
import time
import numpy as np

from agent.alpha_z import AlphaZero
from agent.monte_carlo import MonteCarlo
from alpha_zero.multi_process import predict_process, THREAD_COUNT
from board import PLAYER1, PLAYER2, Board, NO_ONE


def compare(net_name, rounds, simple_mcts_depth):
    pipes = [mp.Pipe() for _ in range(THREAD_COUNT)]

    match_p = mp.Process(target=predict_process, args=(net_name, [pi[0] for pi in pipes]))
    match_p.start()

    times_per_process = math.ceil(rounds / THREAD_COUNT)

    with mp.Pool(processes=THREAD_COUNT)as pool:
        data = pool.starmap(play_against_mcts,
                            [(pi[1], times_per_process, 800, simple_mcts_depth, idx >= (THREAD_COUNT / 2))
                             for idx, pi in enumerate(pipes)])
        pool.close()
        pool.join()

    alpha_net = sum([d[0] for d in data])
    mcts = sum([d[1] for d in data])
    t = sum([d[2] for d in data])
    print(f'AlphaNet:{alpha_net} MCTS:{mcts} Tie:{t}')

    match_p.join()

def play_against_mcts(net, times, mcts_turns, simple_mcts_depth, alpha_first):
    player_1 = AlphaZero(PLAYER1, net,
                         mcts_turns=mcts_turns,
                         multi_process=True)
    player_2 = MonteCarlo(PLAYER2, simple_mcts_depth)

    if alpha_first:
        player_1, player_2 = player_2, player_1
        player_1.player = PLAYER1
        player_2.mcts.player = PLAYER2

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

        winner = board.winner
        if winner == PLAYER1:
            p1 += 1
        elif winner == PLAYER2:
            p2 += 1
        elif winner == NO_ONE:
            t += 1

    net.send('Finished')

    if not alpha_first:
        return p1, p2, t
    else:
        return p2, p1, t


if __name__ == '__main__':
    compare(net_name='final_net_20200522_0150', rounds=10, simple_mcts_depth=300)
