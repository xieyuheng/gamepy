import numpy as np
import pandas as pd

import gamepy as gm

abc_dic = {
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9],
    "I": [0, 1, 2],
    "J": [2, 1, 0],
    "X": [0, 0, 0], # this column will be ignored
}

abc_df = pd.DataFrame (
    abc_dic,
    index = [100, 200, 300])

abc_cube = gm.data_cube_c (
    # "D" and "K" will be ignored
    abc_df, ["A", "B", "C", "D"], ["I", "J", "K"])

def test_abc_cube ():
    assert abc_cube.attr_name_list == ["A", "B", "C"]
    assert abc_cube.indi_name_list == ["I", "J"]
    pd.testing.assert_index_equal (
        abc_cube.df.index,
        pd.Index ([100, 200, 300]))

def test_proj ():
    cube = abc_cube.proj ({
        "A": 1,
        "B": 4,
    })
    assert type (cube) == gm.data_cube_c

def test_proj ():
    cube = abc_cube.proj_drop ({
        "A": 1,
        "B": 4,
    })
    assert cube.attr_name_list == ["C"]
    assert type (cube) == gm.data_cube_c
