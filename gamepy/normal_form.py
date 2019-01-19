import numpy as np
import pandas as pd

# player: str
# action: str
# pure_plan: (player, action)
# pure_plans: pd.Series
# opponent_pure_plans: pd.Series -- length == self.players.size - 1
# pure_plan_profile: pd.Series -- length == self.players.size

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

    def pure_plans_c (self, pure_plans, name = "pure_plans"):
        return pd.Series (
            pure_plans,
            name = name,
            index = self.players.index)

    def pure_plan_to_index (self, pure_plan):
        player, action = pure_plan
        if pd.isna (action):
            return slice (None)
        else:
            return self.players [player] .index (action)

    def pure_plans_to_index_list (self, pure_plans):
        return list (map (
            lambda pure_plan: self.pure_plan_to_index (pure_plan),
            pure_plans.items ()))

    def payoff_array (self, pure_plan_profile):
        pure_plan_profile = self.pure_plans_c (pure_plan_profile)
        assert pure_plan_profile.size == self.players.size
        index_list = self.pure_plans_to_index_list (pure_plan_profile)
        return self.payoff_tensor [tuple (index_list)]

    def payoff_series (self, pure_plan_profile):
        return pd.Series (
            self.payoff_array (pure_plan_profile),
            name = "payoffs",
            index = self.players.index)

    def responses (self, opponent_pure_plans):
        opponent_pure_plans = self.pure_plans_c (opponent_pure_plans)
        miss = opponent_pure_plans [ opponent_pure_plans.isna () ]
        player_list = miss.keys ()
        assert len (player_list) == 1
        player = player_list [0]
        action_list = self.players [player]
        index_list = self.pure_plans_to_index_list (opponent_pure_plans)
        multi_index = pd.MultiIndex.from_arrays (
            [action_list],
            names = [player])
        return pd.DataFrame (
            self.payoff_tensor [tuple (index_list)],
            index = multi_index,
            columns = self.players.keys ())

    # [todo]
    # - design the interface for both pure_pure_plan and mixed_pure_plan

    '''
    - we might can not generalize the implement of normal_form
      to an abstract type of which
      pure and mixed version of the game are its subtype
      because two sets of interface might both be used
    '''

    '''
    - but we might want to generalize the implement
      because extensive-form game might be included
    '''

    # [todo]
    # best_responses instead of best_response

    def best_response (self, opponent_pure_plans):
        responses = self.responses (opponent_pure_plans)
        player = responses.index.names [0]
        return responses.sort_values (
            by = player,
            axis = 0,
            ascending = False) .head (1)

    # def nash_equilibrium (self):
    #     return

    # def dominant (self, pure_plan1, pure_plan2):
    #     return

    # def dominant_pure_plan (self):
    #     return
