import numpy as np
import pandas as pd

from gamepy import tensor_c

def prisoner_s_dilemma ():
    axes = {
        "Alice": [ "silent", "betray" ],
        "Bob": [ "silent", "betray" ],
        "payoff": [ "Alice", "Bob" ],
    }

    ndarray = np.array ([
        [[-1, -1], [-3, -0]],
        [[-0, -3], [-2, -2]],
    ])

    return tensor_c (axes, ndarray)

def test_idx ():
    tensor = prisoner_s_dilemma ()

    np.testing.assert_array_equal (
        tensor.idx ({}),
        np.array ([
            [[-1, -1], [-3, -0]],
            [[-0, -3], [-2, -2]],
        ]))

    np.testing.assert_array_equal (
        tensor.idx ({
            "Alice": "silent",
        }),
        np.array ([
            [-1, -1], [-3, -0],
        ]))

    np.testing.assert_array_equal (
        tensor.idx ({
            "Alice": "silent",
            "Bob": "silent",
        }),
        np.array ([-1, -1]))

    assert tensor.idx ({
        "Alice": "silent",
        "Bob": "silent",
        "payoff": "Alice",
    }) == -1

def test_proj ():
    tensor = prisoner_s_dilemma () .proj ({
        "Alice": "silent"
    })

    np.testing.assert_array_equal (
        tensor.idx ({}),
        np.array ([
            [-1, -1], [-3, -0],
        ]))

    np.testing.assert_array_equal (
        tensor.idx ({
            "Bob": "silent",
        }),
        np.array ([-1, -1]))

    assert tensor.idx ({
        "Bob": "silent",
        "payoff": "Alice",
    }) == -1
