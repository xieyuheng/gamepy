import numpy as np
import pandas as pd

from gamepy import normal_form

def prisoner_s_dilemma():
    action_list = [
        "silent",
        "betray",
    ]

    players = {
        "Alice": action_list,
        "Bob": action_list,
    }

    payoff_tensor = np.array([
        [[-1, -1], [-3, -0]],
        [[-0, -3], [-2, -2]],
    ])

    return normal_form.c(players, payoff_tensor)

def test_payoff_array():
    game = prisoner_s_dilemma()

    # strategy_profile: pd.Series like

    np.testing.assert_array_equal(
        game.payoff_array({
            "Alice": "silent",
            "Bob": "silent",
        }),
        np.array([-1, -1]))

    np.testing.assert_array_equal(
        game.payoff_array({
            "Alice": "silent",
            "Bob": "betray",
        }),
        np.array([-3, -0]))

    np.testing.assert_array_equal(
        game.payoff_array({
            "Alice": "betray",
            "Bob": "silent",
        }),
        np.array([-0, -3]))

    np.testing.assert_array_equal(
        game.payoff_array({
            "Alice": "betray",
            "Bob": "betray",
        }),
        np.array([-2, -2]))

def test_payoff_series():
    game = prisoner_s_dilemma()

    payoff_series = game.payoff_series({
        "Alice": "silent",
        "Bob": "betray",
    })
    assert payoff_series["Alice"] == -3

    # the order of dict does not matter

    payoff_series = game.payoff_series({
        "Bob": "betray",
        "Alice": "silent",
    })
    assert payoff_series["Alice"] == -3

def test_responses():
    game = prisoner_s_dilemma()

    game.responses({
        "Bob": "betray",
    })
