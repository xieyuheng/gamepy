import numpy as np
import pandas as pd

import gamepy as gm

abc_dic = {
    "A": [1, 0, 2, 8, 3],
    "B": [4, 4, 5, 5, 6],
    "C": [7, 7, 8, 8, 9],
    "I": [0, 1, 1, 1, 2],
    "J": [2, 3, 1, 1, 0],
    "X": [0, 0, 0, 0, 0], # this column will be ignored
}

abc_df = pd.DataFrame (
    abc_dic,
    index = [1000, 1000, 2000, 2000, 3000])

abc_cube = gm.data_cube_c (
    # "D" and "K" will be ignored
    abc_df, ["A", "B", "C", "D"], ["I", "J", "K"])

def test_abc_cube ():
    print ("<test_abc_cube>")
    print ("abc_cube = {}" .format (abc_cube))
    print ("</test_abc_cube>")
    assert abc_cube.attr_name_list == ["A", "B", "C"]
    assert abc_cube.indi_name_list == ["I", "J"]

def test_proj ():
    cube = abc_cube.proj ({
        "B": 4,
        "C": 7,
    })
    print ("<test_proj>")
    print ("cube = {}" .format (cube))
    print ("</test_proj>")
    assert cube.attr_name_list == ["A", "B", "C"]
    assert cube.indi_name_list == ["I", "J"]
    assert type (cube) == gm.data_cube_c

def test_proj_drop ():
    cube = abc_cube.proj_drop ({
        "B": 4,
        "C": 7,
    })
    print ("<test_proj_drop>")
    print ("cube = {}" .format (cube))
    print ("</test_proj_drop>")
    assert cube.attr_name_list == ["A"]
    assert cube.indi_name_list == ["I", "J"]
    assert type (cube) == gm.data_cube_c

def test_retract ():
    cube = abc_cube.retract (["A"])
    print ("<test_retract>")
    print ("cube = {}" .format (cube))
    print ("</test_retract>")
    assert cube.attr_name_list == ["B", "C"]
    assert cube.indi_name_list == ["I", "J"]
    assert type (cube) == gm.data_cube_c
