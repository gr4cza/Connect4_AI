from board import Board, PLAYER1, PLAYER2
from random_agent import RandomAgent


def play_game():
    board = Board()

    while True:
        play_as = input("Choose token! (\"O\"/\"X\")? Note: \"O\" starts \n")
        if play_as.lower() in ["o", "x"]:
            if play_as.lower() == "o":
                player = PLAYER1
                break
            elif play_as.lower() == "x":
                player = PLAYER2
                break
        else:
            print("Not a valid choice!")

    while board.check_win() == 0 and board.available_moves() != []:
        if player == PLAYER1:
            print(board)
            while True:
                col = input(f"Choose a column? {[x + 1 for x in board.available_moves()]}\n")
                if board.add_token(int(col) - 1):
                    player = PLAYER2
                    break
        elif player == PLAYER2:
            col = RandomAgent.move(board)
            print(f'The machine choose: {col}')
            board.add_token(col)
            player = PLAYER1

    print('-' * 30)
    print(board)
    if board.check_win() != 0:
        print(f'{"O" if board.check_win() == 1 else "X"} wins!')
    else:
        print('Tie!')


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