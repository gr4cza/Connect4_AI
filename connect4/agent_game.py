from agent.min_max import MinMaxAgentWAlphaBeta
from agent.random_agent import RandomAgent
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
    print(board)
    print(board.moves)
    return board.get_winner()


def play_n_game(n, player1, player2):
    p1, p2, d = 0, 0, 0
    for i in range(n):
        print(i)
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
    # q_agent = QLearn(PLAYER1, source_name='models/_20191110_162542.pkl')

    # # random game
    # play_n_game(10_000, RandomAgent(), RandomAgent())
    #
    # # min-max basic first vs random
    # play_n_game(10, MinMaxAgentWAlphaBeta(5, basic_score.score, PLAYER1), RandomAgent())
    # # min-max basic second vs random
    # play_n_game(10, RandomAgent(), MinMaxAgentWAlphaBeta(5, basic_score.score, PLAYER2))

    # min-max adv first vs random
    play_n_game(100, MinMaxAgentWAlphaBeta(3, adv_score.score, PLAYER1), RandomAgent())
    # min-max adv second vs random
    play_n_game(100, RandomAgent(), MinMaxAgentWAlphaBeta(3, adv_score.score, PLAYER2))

    # min-max vs min-max adv
    # play_n_game(1, MinMaxAgentWAlphaBeta(5, adv_score.score, PLAYER1),
    #             MinMaxAgentWAlphaBeta(5, adv_score.score, PLAYER2))

    # play_n_game(1000, q_agent, RandomAgent())
    # play_n_game(1, q_agent, MinMaxAgentWAlphaBeta(5, adv_score.score, PLAYER2))
