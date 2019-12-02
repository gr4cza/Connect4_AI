from agent.min_max import MinMaxAgentWAlphaBeta
from agent.monte_carlo import MonteCarlo
from agent.player import Player
from board import Board, PLAYER1, PLAYER2
from evaluator import AdvancedScore


def play_game():
    board = Board()
    score = AdvancedScore()
    p = Player()
    p2 = Player()
    player = PLAYER1
    ai_1 = MinMaxAgentWAlphaBeta(6, score.score)
    ai_2 = MonteCarlo(10_000)

    p2 = ai_2

    p.choose_token()
    p2.player = PLAYER2 if p.player == PLAYER1 else PLAYER1

    while not board.is_game_over():
        if player == PLAYER1:
            choose(board, player, p, p2)
            player = PLAYER2
        elif player == PLAYER2:
            choose(board, player, p, p2)
            player = PLAYER1

    print('-' * 30)
    print(board)
    print(board.moves)
    if board.get_winner() != 0:
        print(f'{"O" if board.get_winner() == PLAYER1 else "X"} wins!')
    else:
        print('Draw!')


def choose(board, player, p1, p2):
    if player == p1.player:
        col = p1.move(board)
        board.add_token(col)
    else:
        col = p2.move(board)
        print(f'The machine choose: {col + 1}')
        board.add_token(col)


if __name__ == '__main__':
    play_again = True
    while play_again:
        play_game()
        while True:
            again = input("Do you wanna play again? (Y/N)\n")
            if again.lower() in ["y", "n"]:
                if again.lower() == "n":
                    play_again = False
                break
