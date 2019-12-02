from board import PLAYER1, PLAYER2, NO_ONE


class Player(object):
    def __init__(self):
        self.player = NO_ONE

    def choose_token(self):
        while True:
            play_as = input('Choose token! ("O"/"X")? Note: "O" starts \n')
            if play_as.lower() in ["o", "x"]:
                if play_as.lower() == "o":
                    self.player = PLAYER1
                    break
                elif play_as.lower() == "x":
                    self.player = PLAYER2
                    break
            else:
                print("Not a valid choice!")

    @staticmethod
    def move(board):
        print(board)
        while True:
            col = input(f"Choose a column? {[x + 1 for x in board.available_moves()]}\n")
            break
        return int(col) - 1
