from multiprocessing import Process
from time import strftime

from alpha_zero.game_data import GameData
from alpha_zero.multi_process import multi_self_play, evaluate, train_net_process


def train(turns, hours=0., minutes=0., mcts_turns=300, epochs=10, net_name=None, source_net=None, train_first=False):
    print(f'Training started with turns={turns}, hours={hours}, minutes={minutes}, mcts_turns={mcts_turns}, '
          f'net_name={net_name}, source_net={source_net}, train_first={train_first}\n'
          f'Time:{strftime("%Y.%m.%d. %H:%M")}')

    # variables
    if source_net is not None:
        net_name = source_net
        print(f'Continuing train from {source_net}')
    else:
        net_name = net_name + f'_{strftime("%Y%m%d_%H%M")}'
        print(f'New training with {net_name}')
    game_data = GameData(source_net)

    for i in range(turns):
        if not train_first:
            # self play
            data_run = multi_self_play(net_name=net_name, hours=hours, minutes=minutes, mcts_turns=mcts_turns)

            # add new game_data to database
            game_data.add_games(data_run)

            # save new database
            game_data.save(net_name)

        print(game_data)

        # load net & train
        p = Process(target=train_net_process, args=(net_name, epochs))
        p.start()
        p.join()

        # evaluate
        evaluate(net_name, 100, mcts_turns)

        train_first = False


if __name__ == '__main__':
    train(net_name='test', turns=3, minutes=5, mcts_turns=20, epochs=3)
