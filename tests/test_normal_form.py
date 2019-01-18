import numpy as np
import pandas as pd

from gamepy import normal_form

def prisoner_s_dilemma ():
    action_list = [
        "silent",
        "betray",
    ]

    players = {
        "Alice": action_list,
        "Bob": action_list,
    }

    payoff_tensor = np.array ([
        [[-1, -1], [-3, -0]],
        [[-0, -3], [-2, -2]],
    ])

    return normal_form.c (players, payoff_tensor)

def test_payoff_array ():
    game = prisoner_s_dilemma ()

    # strategy_profile: pd.Series like

    np.testing.assert_array_equal (
        game.payoff_array ({
            "Alice": "silent",
            "Bob": "silent",
        }),
        np.array ([-1, -1]))

    np.testing.assert_array_equal (
        game.payoff_array ({
            "Alice": "silent",
            "Bob": "betray",
        }),
        np.array ([-3, -0]))

    np.testing.assert_array_equal (
        game.payoff_array ({
            "Alice": "betray",
            "Bob": "silent",
        }),
        np.array ([-0, -3]))

    np.testing.assert_array_equal (
        game.payoff_array ({
            "Alice": "betray",
            "Bob": "betray",
        }),
        np.array ([-2, -2]))

def test_payoff_series ():
    game = prisoner_s_dilemma ()

    payoff_series = game.payoff_series ({
        "Alice": "silent",
        "Bob": "betray",
    })
    assert payoff_series ["Alice"] == -3

    # the order of dict does not matter

    payoff_series = game.payoff_series ({
        "Bob": "betray",
        "Alice": "silent",
    })
    assert payoff_series ["Alice"] == -3

def test_responses ():
    game = prisoner_s_dilemma ()

    responses = game.responses ({
        "Alice": "silent",
    })
    np.testing.assert_array_equal (
        responses.values,
        np.array ([
            [-1, -1], [-3, -0],
        ]))
    print (responses)

    responses = game.responses ({
        "Bob": "betray",
    })
    np.testing.assert_array_equal (
        responses.values,
        np.array ([
            [-3, -0],
            [-2, -2],
        ]))
    print (responses)

def test_bast_response ():
    game = prisoner_s_dilemma ()

    response = game.bast_response ({
        "Alice": "silent",
    })
    np.testing.assert_array_equal (
        response.values,
        np.array ([[-3, -0]]))

    response = game.bast_response ({
        "Bob": "betray",
    })
    np.testing.assert_array_equal (
        response.values,
        np.array ([[-2, -2]]))
