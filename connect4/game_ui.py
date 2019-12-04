import math
import sys

import pygame

from agent.monte_carlo import MonteCarlo
from board import Board, PLAYER1, PLAYER2, NO_ONE

SPACING = 2

RED = (220, 0, 0)

YELLOW = (238, 219, 4)

BACKGROUND = (255, 255, 255)
BLUE = (0, 123, 255)

TOKEN_SIZE = 100
WIDTH = 7 * (TOKEN_SIZE + SPACING)
HEIGHT = 7 * TOKEN_SIZE


class GameUI(object):
    def __init__(self, p1, p2) -> None:
        self.p2 = p2
        self.p1 = p1
        self.p1.player, self.p2.player = PLAYER1, PLAYER2
        self.game_over = False
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    def game_loop(self):
        board = Board()
        turn = PLAYER1
        self._draw_board(board)
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            if turn == PLAYER1:
                self._choose(board, turn)
                turn = PLAYER2
            elif turn == PLAYER2:
                self._choose(board, turn)
                turn = PLAYER1
            if board.is_game_over():
                pygame.time.wait(5000)
                self.game_over = True

    def _draw_board(self, board):
        radius = int(TOKEN_SIZE / 2)
        b = board.board
        row, column = b.shape
        pygame.draw.rect(self.screen, BLUE, (0, HEIGHT - row * TOKEN_SIZE, WIDTH, row * TOKEN_SIZE))

        for r in range(row):
            for c in range(column):
                pygame.draw.circle(self.screen, BACKGROUND,
                                   (c * (TOKEN_SIZE + SPACING) + radius, (r + 1) * TOKEN_SIZE + radius),
                                   (radius - 4))
                token_color = BACKGROUND
                if b[r][c] == PLAYER1:
                    token_color = YELLOW
                elif b[r][c] == PLAYER2:
                    token_color = RED

                pygame.draw.circle(self.screen, token_color,
                                   (c * (TOKEN_SIZE + SPACING) + radius, (r + 1) * TOKEN_SIZE + radius),
                                   (radius - 4))
        pygame.display.update()

    def _choose(self, board, turn):
        if turn == self.p1.player:
            col = self.p1.move(board)
            board.add_token(col)
        else:
            col = self.p2.move(board)
            board.add_token(col)
        self._draw_board(board)


class PlayerUI(object):
    def __init__(self) -> None:
        self.player = NO_ONE

    @staticmethod
    def move(board):
        col = None
        legal_moves = board.available_moves()
        choose = False
        screen = pygame.display.get_surface()
        while not choose:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BACKGROUND, (0, 0, WIDTH, TOKEN_SIZE))
                    posx = event.pos[0]
                    color = YELLOW if board.current_player == PLAYER1 else RED
                    pygame.draw.circle(screen, color, (posx, int(TOKEN_SIZE / 2)), int(TOKEN_SIZE / 2) - 2)
                    pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    column = int(math.floor(event.pos[0] / (TOKEN_SIZE + SPACING)))
                    if column in legal_moves:
                        col = column
                        choose = True
                        pygame.draw.rect(screen, BACKGROUND, (0, 0, WIDTH, TOKEN_SIZE))
        return col


if __name__ == '__main__':
    game = GameUI(MonteCarlo(5_000), MonteCarlo(10_000))
    game.game_loop()
