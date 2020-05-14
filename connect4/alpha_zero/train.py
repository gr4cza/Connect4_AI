from alpha_zero.game_data import GameData
from alpha_zero.multi_process import multi_self_play


def train(times):
    # variables
    best_net_name = 'test_net_1'
    data = GameData('test_0')

    for i in range(times):
        # self play
        data_run = multi_self_play(net_name=best_net_name, n=10, mcts_turns=20)

        # add new data to database
        data.add_games(data_run)

        # save new database
        data.save(f'test_{i}')

        # load net
        from alpha_zero.alpha_net import AlphaNet
        best_net = AlphaNet(best_net_name)

        # retrain
        new_net = best_net.train(data, epochs=3, new_name=f'{best_net_name}_{i + 1}')

        # close net
        best_net.release()

        # evaluate
        best_net_name = evaluate(best_net_name, new_net)


def evaluate(net_1, net_2):
    return net_2  # TODO


if __name__ == '__main__':
    train(3)
