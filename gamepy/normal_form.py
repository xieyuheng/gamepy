import numpy as np
import pandas as pd

# player: str
# action: str
# strategy: (player, action)
# strategies: pd.Series
# opponent_strategies: pd.Series -- length == self.players.size - 1
# strategy_profile: pd.Series -- length == self.players.size

class c:
    def __init__(self, players, payoff_tensor):
        self.players = pd.Series(players, name="players")
        self.player_shape = tuple(map(len, self.players.values))
        self.payoff_tensor_shape = (*self.player_shape, len(players))

        assert payoff_tensor.shape == self.payoff_tensor_shape
        self.payoff_tensor = payoff_tensor

    def Strategies(self, strategies, name="Strategies"):
        return pd.Series(
            strategies,
            name=name,
            index=self.players.index,
        ).dropna()

    def strategy_index(self, strategy):
        player, action = strategy
        return self.players[player].index(action)

    def payoff_array(self, strategy_profile):
        strategy_profile = self.Strategies(strategy_profile)
        assert strategy_profile.size == self.players.size
        index_list = map(
            lambda strategy: self.strategy_index(strategy),
            strategy_profile.items())
        return self.payoff_tensor[tuple(index_list)]

    def payoff_series(self, strategy_profile):
        return pd.Series(
            self.payoff_array(strategy_profile),
            name="playoffs",
            index=self.players.index)

    # def bast_response_p(self, strategy, opponent_strategies):
    #     return

    def responses(self, opponent_strategies):
        opponent_strategies = self.Strategies(opponent_strategies)
        assert opponent_strategies.size == self.players.size - 1
        player = set(self.players.index) - set(opponent_strategies.index)
        action_list = self.players[player]
        print ("player = {}".format(player))

    def bast_response(self, opponent_strategies):
        return

    # def opponent_strategies(self, strategy):
    #     return

    # def nash_equilibrium(self):
    #     return

    # def dominant_strategy(self):
    #     return
