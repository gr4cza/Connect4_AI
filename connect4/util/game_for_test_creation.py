from board import Board


def play_game():
    board = Board()
    moves = []
    while board.get_winner() == 0 and board.available_moves() != []:
        col = input()
        if col == "b":
            break
        board.add_token(int(col) - 1)
        moves.append(int(col) - 1)
        print(board)

    print('-' * 30)
    print(board)
    print(moves)


if __name__ == '__main__':
    play_game()
