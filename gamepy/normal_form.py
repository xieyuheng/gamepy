import numpy as np
import pandas as pd

# player: str
# action: str
# pure_strategy_item: (player, action)
# pure_strategy: pd.Series
# opponent_pure_strategy: pd.Series -- length == self.players.size - 1
# pure_strategy_profile: pd.Series -- length == self.players.size

class c:
    def __init__ (self, players, payoff_tensor):
        self.players = pd.Series (players, name = "players")
        self.player_shape = tuple (map (len, self.players.values))
        self.payoff_tensor_shape = tuple ([
            *self.player_shape,
            len (players),
        ])
        assert payoff_tensor.shape == self.payoff_tensor_shape
        self.payoff_tensor = payoff_tensor

    def pure_strategy_c (self, pure_strategy, name = "pure_strategy"):
        return pd.Series (
            pure_strategy,
            name = name,
            index = self.players.index)

    def pure_strategy_item_to_index (self, pure_strategy):
        player, action = pure_strategy
        if pd.isna (action):
            return slice (None)
        else:
            return self.players [player] .index (action)

    def pure_strategy_to_index_list (self, pure_strategy):
        return list (map (
            lambda pure_strategy: self.pure_strategy_item_to_index (pure_strategy),
            pure_strategy.items ()))

    def pure_payoff_array (self, pure_strategy_profile):
        pure_strategy_profile = self.pure_strategy_c (pure_strategy_profile)
        assert pure_strategy_profile.size == self.players.size
        index_list = self.pure_strategy_to_index_list (pure_strategy_profile)
        return self.payoff_tensor [tuple (index_list)]

    def pure_payoff_series (self, pure_strategy_profile):
        return pd.Series (
            self.pure_payoff_array (pure_strategy_profile),
            name = "payoffs",
            index = self.players.index)

    def pure_responses (self, opponent_pure_strategy):
        opponent_pure_strategy = self.pure_strategy_c (opponent_pure_strategy)
        miss = opponent_pure_strategy [ opponent_pure_strategy.isna () ]
        player_list = miss.keys ()
        assert len (player_list) == 1
        player = player_list [0]
        action_list = self.players [player]
        index_list = self.pure_strategy_to_index_list (opponent_pure_strategy)
        multi_index = pd.MultiIndex.from_arrays (
            [action_list],
            names = [player])
        return pd.DataFrame (
            self.payoff_tensor [tuple (index_list)],
            index = multi_index,
            columns = self.players.keys ())

    def best_pure_responses (self, opponent_pure_strategy):
        pure_responses = self.pure_responses (opponent_pure_strategy)
        player = pure_responses.index.names [0]
        grouped_responses = pure_responses.groupby (player)
        _value, df = list (grouped_responses) [-1]
        return df

    # def pure_nash_equilibrium (self):
    #     return

    # def pure_dominant (self, pure_strategy1, pure_strategy2):
    #     return

    # def dominant_pure_strategy (self):
    #     return
