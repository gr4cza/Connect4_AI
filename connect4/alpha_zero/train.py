from agent.alpha_z import AlphaZero
from alpha_zero.alpha_net import AlphaNet
from board import PLAYER1, PLAYER2, Board


def self_play(n=10):
    alpha_net = AlphaNet()
    az_1 = AlphaZero(PLAYER1, alpha_net)
    az_2 = AlphaZero(PLAYER2, alpha_net)

    for i in range(n):
        print('selfplay {} starting'.format(i))
        board = Board()
        current_player = PLAYER1

        while not board.is_game_over():
            if current_player == PLAYER1:
                action = az_1.move(board)
                board.add_token(action)
                current_player = PLAYER2
            elif current_player == PLAYER2:
                action = az_2.move(board)
                board.add_token(action)
                current_player = PLAYER1

        print(board.winner)


if __name__ == '__main__':
    self_play(3)
