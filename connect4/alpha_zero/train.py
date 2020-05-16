from time import strftime

from alpha_zero.game_data import GameData
from alpha_zero.multi_process import multi_self_play, evaluate


def train(turns, hours, mcts_turns, epochs, net_name=None, source_net=None):
    # variables
    if net_name is None and source_net is None:
        net_name = 'test'

    from_source = False
    if source_net and net_name is not None:
        from_source = True

    if source_net and net_name is None:
        net_name = source_net
    else:
        net_name = net_name + f'_{strftime("%Y%m%d_%H%M")}'
    data = GameData(source_net)

    for i in range(turns):
        # self play
        if not from_source:
            data_run = multi_self_play(net_name=net_name, hours=hours, mcts_turns=mcts_turns)
        else:
            data_run = multi_self_play(net_name=source_net, hours=hours, mcts_turns=mcts_turns)

        # add new data to database
        data.add_games(data_run)

        # save new database
        data.save(net_name)

        # load net
        from alpha_zero.alpha_net import AlphaNet
        if not from_source:
            loaded_net = AlphaNet(net_name)
        else:
            loaded_net = AlphaNet(source_net)

        # retrain
        if not from_source:
            loaded_net.train(data, epochs=epochs)
        else:
            loaded_net.train(data, epochs=epochs, new_model_name=net_name)

        # close net
        loaded_net.release()

        # evaluate
        if not from_source:
            evaluate(net_name)
        else:
            from_source = evaluate(net_name, source_net)


if __name__ == '__main__':
    train(turns=3, hours=0.1, mcts_turns=20, epochs=3)
