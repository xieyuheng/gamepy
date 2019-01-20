import numpy as np
import pandas as pd

import gamepy as gm

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

    return gm.normal_form_c (players, payoff_tensor)

def test_pure_payoff_array ():
    game = prisoner_s_dilemma ()

    np.testing.assert_array_equal (
        game.proj ({
            "Alice": "silent",
            "Bob": "silent",
        }) .tensor (),
        np.array ([-1, -1]))

    np.testing.assert_array_equal (
        game.proj ({
            "Alice": "silent",
            "Bob": "betray",
        }) .tensor (),
        np.array ([-3, -0]))

    np.testing.assert_array_equal (
        game.proj ({
            "Alice": "betray",
            "Bob": "silent",
        }) .tensor (),
        np.array ([-0, -3]))

    np.testing.assert_array_equal (
        game.proj ({
            "Alice": "betray",
            "Bob": "betray",
        }) .tensor (),
        np.array ([-2, -2]))

def test_pure_responses ():
    game = prisoner_s_dilemma ()

    pure_responses = game.pure_responses ({
        "Alice": "silent",
    })
    np.testing.assert_array_equal (
        pure_responses.values,
        np.array ([
            [-1, -1], [-3, -0],
        ]))

    pure_responses = game.pure_responses ({
        "Bob": "betray",
    })
    np.testing.assert_array_equal (
        pure_responses.values,
        np.array ([
            [-3, -0],
            [-2, -2],
        ]))

def test_best_pure_responses ():
    game = prisoner_s_dilemma ()

    pure_responses = game.best_pure_responses ({
        "Alice": "silent",
    })
    np.testing.assert_array_equal (
        pure_responses.values,
        np.array ([[-3, -0]]))

    pure_responses = game.best_pure_responses ({
        "Bob": "betray",
    })
    np.testing.assert_array_equal (
        pure_responses.values,
        np.array ([[-2, -2]]))
