from agent.min_max import MinMaxAgentWAlphaBeta
from board import Board, PLAYER1, PLAYER2
from evaluator import AdvancedScore


def play_game():
    board = Board()
    score = AdvancedScore()

    while True:
        play_as = input('Choose token! ("O"/"X")? Note: "O" starts \n')
        if play_as.lower() in ["o", "x"]:
            if play_as.lower() == "o":
                player = PLAYER1
                ai = MinMaxAgentWAlphaBeta(7, score.score, PLAYER2)
                break
            elif play_as.lower() == "x":
                player = PLAYER2
                ai = MinMaxAgentWAlphaBeta(7, score.score, PLAYER1)
                break
        else:
            print("Not a valid choice!")

    while not board.is_game_over():
        if player == PLAYER1:
            print(board)
            while True:
                col = input(f"Choose a column? {[x + 1 for x in board.available_moves()]}\n")
                if board.add_token(int(col) - 1):
                    player = PLAYER2
                    break
        elif player == PLAYER2:
            col = ai.move(board)
            print(f'The machine choose: {col + 1}')
            board.add_token(col)
            player = PLAYER1

    print('-' * 30)
    print(board)
    print(board.moves)
    if board.get_winner() != 0:
        print(f'{"O" if board.get_winner() == 1 else "X"} wins!')
    else:
        print('Draw!')


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
