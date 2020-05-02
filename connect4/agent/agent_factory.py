from enum import Enum

from agent.min_max.evaluator import AdvancedScore
from agent.min_max.min_max import MinMaxAgentWAlphaBeta, MinMaxAgent
from agent.monte_carlo import MonteCarlo
from agent.player import Player
from agent.q_learn import QLearn
from agent.random_agent import RandomAgent
from board import PLAYER1, PLAYER2


class AgentFactory:

    def get_agent_1(self, agent_type):
        return self._get_agent(agent_type, PLAYER1)

    def get_agent_2(self, agent_type):
        return self._get_agent(agent_type, PLAYER2)

    @staticmethod
    def _get_agent(agent_type, player):
        if agent_type == 'Player':
            return Player()
        if agent_type == 'RandomPlayer':
            return RandomAgent()
        if agent_type == 'QLearn':
            return QLearn(player, source_name='models/min_max_5_10K_p1_20191111_202358.pkl')
        if agent_type == 'MinMax':
            return MinMaxAgent(player, 4, AdvancedScore.score)
        if agent_type == 'AlphaBeta':
            return MinMaxAgentWAlphaBeta(player, 6, AdvancedScore.score)
        if agent_type == 'MonteCarlo':
            return MonteCarlo(player, 2000)
