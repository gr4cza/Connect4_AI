from agent.min_max import MinMaxAgentWAlphaBeta
from agent.q_learn import QLearnTrain
from board import Board, PLAYER1, PLAYER2
from evaluator import BasicScore, AdvancedScore


def play_game(player1, player2):
    board = Board()
    player = PLAYER1

    while not board.is_game_over():
        if player == PLAYER1:
            col = player1.move(board)
            board.add_token(col)
            player = PLAYER2
        elif player == PLAYER2:
            col = player2.move(board)
            board.add_token(col)
            player = PLAYER1
    # print(board)
    return board.get_winner()


def play_n_game(n, player1, player2):
    p1, p2, d = 0, 0, 0
    for i in range(n):
        winner = play_game(player1, player2)
        if winner == 0:
            d += 1
        elif winner == PLAYER1:
            p1 += 1
        elif winner == PLAYER2:
            p2 += 1

    print(f'p1: {p1}, p2: {p2}, d: {d}')


if __name__ == '__main__':
    adv_score = AdvancedScore()
    basic_score = BasicScore()
    # train = QLearnTrain(PLAYER1)
    # train.learn(10000)
    # q_agent = QLearn(PLAYER1, source_name='models/min_max_5_10K_p1_20191111_202358.pkl')

    # # random game
    # play_n_game(10_000, RandomAgent(), RandomAgent())
    #
    # # min-max basic first vs random
    # play_n_game(10, MinMaxAgentWAlphaBeta(5, basic_score.score, PLAYER1), RandomAgent())
    # # min-max basic second vs random
    # play_n_game(10, RandomAgent(), MinMaxAgentWAlphaBeta(5, basic_score.score, PLAYER2))

    # # min-max adv first vs random
    # play_n_game(100, MinMaxAgentWAlphaBeta(3, adv_score.score, PLAYER1), RandomAgent())
    # # min-max adv second vs random
    # play_n_game(100, RandomAgent(), MinMaxAgentWAlphaBeta(3, adv_score.score, PLAYER2))

    # min-max vs min-max adv
    # play_n_game(1, MinMaxAgentWAlphaBeta(5, adv_score.score, PLAYER1),
    #             MinMaxAgentWAlphaBeta(5, adv_score.score, PLAYER2))

    # play_n_game(100_000, q_agent, RandomAgent())
    # play_n_game(100_000, RandomAgent(), RandomAgent())
    # play_n_game(1, q_agent, MinMaxAgentWAlphaBeta(5, adv_score.score, PLAYER2))
    train = QLearnTrain(PLAYER1)
    train.learn(100_000, against=MinMaxAgentWAlphaBeta(4, adv_score.score, PLAYER2), name=f'min_max_4_100K_p1')
    train.player = PLAYER2
    train.learn(100_000, against=MinMaxAgentWAlphaBeta(4, adv_score.score, PLAYER1), name=f'min_max_4_100K_both')
    train.reset()
