class Player(object):

    @staticmethod
    def move(board):
        print(board)
        while True:
            a_moves = board.available_moves()
            col = input(f"Choose a column! {[x + 1 for x in board.available_moves()]}\n")
            col = col if col.isdigit() else -1
            col = int(col)
            if col-1 in a_moves:
                break
            else:
                print('Invalid input')
        return col - 1
