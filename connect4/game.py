from agent.agent_factory import AgentFactory
from board import Board, PLAYER1, PLAYER2, NO_ONE

ENEMY = 'AlphaZero'


def play_game():
    board = Board()
    current_player = PLAYER1
    factory = AgentFactory()

    human_player_token = choose_token()
    if human_player_token == PLAYER1:
        p1 = factory.get_agent_1('Player')
        p2 = factory.get_agent_2(ENEMY)
    else:
        p1 = factory.get_agent_1(ENEMY)
        p2 = factory.get_agent_2('Player')

    while not board.is_game_over():
        if current_player == PLAYER1:
            choose(board, p1)
            current_player = PLAYER2
        elif current_player == PLAYER2:
            choose(board, p2)
            current_player = PLAYER1

    print('-' * 30)
    print(board)

    if board.winner != NO_ONE:
        print(f'{"O" if board.winner == PLAYER1 else "X"} wins!')
    else:
        print('Draw!')


def choose(board, player):
    col = player.move(board)
    board.add_token(col)


def choose_token():
    while True:
        play_as = input('Choose token! ("O"/"X")? Note: "O" starts \n')
        if play_as.lower() in ["o", "x"]:
            if play_as.lower() == "o":
                return PLAYER1
            elif play_as.lower() == "x":
                return PLAYER2
        else:
            print("Not a valid choice!")


if __name__ == '__main__':
    play_again = True
    while play_again:
        play_game()
        while True:
            again = input('Do you wanna play again? (Y/N)\n')
            if again.lower() in ['y', 'n']:
                if again.lower() == 'n':
                    play_again = False
                break
