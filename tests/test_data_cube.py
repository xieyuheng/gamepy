import numpy as np
import pandas as pd

import gamepy as gm

def test_data_cube_c():
    dic = {
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    }

    df = pd.DataFrame (dic, index = [6, 6, 6])
    cube = gm.data_cube_c (df, ["A"], ["B"])

    assert cube.attr_name_list == ["A"]
    assert cube.indi_name_list == ["B"]
    pd.testing.assert_index_equal (
        cube.index,
        pd.Index ([6, 6, 6]))

    cube = gm.data_cube_c (
        dic,
        attr_name_list = ["A"],
        indi_name_list = ["B"],
        index = [6, 6, 6])

    assert cube.attr_name_list == ["A"]
    assert cube.indi_name_list == ["B"]
    pd.testing.assert_index_equal (
        cube.index,
        pd.Index ([6, 6, 6]))

    cube = gm.data_cube_c (
        dic, ["A"], ["B"],
        index = [6, 6, 6])

    assert cube.attr_name_list == ["A"]
    assert cube.indi_name_list == ["B"]
    pd.testing.assert_index_equal (
        cube.index,
        pd.Index ([6, 6, 6]))
