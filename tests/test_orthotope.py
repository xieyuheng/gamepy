import numpy as np
import pandas as pd

import gamepy as gm

prisoner_s_axes = {
    "Alice": [ "silent", "betray" ],
    "Bob": [ "silent", "betray" ],
    "payoff": [ "Alice", "Bob" ],
}

def prisoner_s_dilemma ():
    tensor = np.array ([
        [[-1, -1], [-3, -0]],
        [[-0, -3], [-2, -2]],
    ])

    return gm.orthotope_c (prisoner_s_axes, tensor)

def test_zeros ():
    np.testing.assert_array_equal (
        gm.ot.zeros (prisoner_s_axes) .tensor (),
        np.array ([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ]))

def test_full ():
    np.testing.assert_array_equal (
        gm.ot.full (prisoner_s_axes, 666) .tensor (),
        np.array ([
            [[666, 666], [666, 666]],
            [[666, 666], [666, 666]],
        ]))

def test_proj_to_tensor ():
    data = prisoner_s_dilemma ()

    np.testing.assert_array_equal (
        data.proj_to_tensor ({}),
        np.array ([
            [[-1, -1], [-3, -0]],
            [[-0, -3], [-2, -2]],
        ]))

    np.testing.assert_array_equal (
        data.proj_to_tensor ({
            "Alice": "silent",
        }),
        np.array ([
            [-1, -1], [-3, -0],
        ]))

    np.testing.assert_array_equal (
        data.proj_to_tensor ({
            "Alice": "silent",
            "Bob": "silent",
        }),
        np.array ([-1, -1]))

    assert data.proj_to_tensor ({
        "Alice": "silent",
        "Bob": "silent",
        "payoff": "Alice",
    }) == -1

def test_proj ():
    data = prisoner_s_dilemma () .proj ({
        "Alice": "silent",
    })

    np.testing.assert_array_equal (
        data.proj_to_tensor ({}),
        np.array ([
            [-1, -1], [-3, -0],
        ]))

    np.testing.assert_array_equal (
        data.proj_to_tensor ({
            "Bob": "silent",
        }),
        np.array ([-1, -1]))

    assert data.proj_to_tensor ({
        "Bob": "silent",
        "payoff": "Alice",
    }) == -1

def test_let ():
    data = prisoner_s_dilemma ()

    data.let ({
        "Alice": "silent",
    }, [
        [-0, -0], [-0, -0]
    ])

    data.let ({
        "Alice": "silent",
        "Bob": "silent",
        "payoff": "Alice",
    }, -9)

    np.testing.assert_array_equal (
        data.proj_to_tensor ({}),
        np.array ([
            [[-9, -0], [-0, -0]],
            [[-0, -3], [-2, -2]],
        ]))

def test_to_data_frame ():
    data = prisoner_s_dilemma () .proj ({
        "Alice": "silent",
    })

    pd.testing.assert_frame_equal (
        data.to_data_frame (),
        pd.DataFrame (
            [[-1, -1], [-3, -0]],
            index = pd.Index (["silent", "betray"], name = "Bob"),
            columns = ["Alice", "Bob"]))

def test_to_series ():
    data = prisoner_s_dilemma () .proj ({
        "Alice": "silent",
        "Bob": "silent",
    })

    pd.testing.assert_series_equal (
        data.to_series (),
        pd.Series (
            [-1, -1],
            index = pd.Index (["Alice", "Bob"], name = "payoff")))
