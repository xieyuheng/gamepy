import numpy as np
import pandas as pd

class data_cube_c (pd.DataFrame):
    @property
    def _constructor (self):
        return data_cube_c

    _metadata = [
        "attr_name_list",
        "indi_name_list",
    ]

    def __init__ (self, df, attr_name_list, indi_name_list, **kwargs):
        assert type (attr_name_list) == list
        assert type (indi_name_list) == list

        super () .__init__ (df, **kwargs)
        self.attr_name_list = attr_name_list
        self.indi_name_list = indi_name_list

        column_name_set = set (self.columns)
        assert set (attr_name_list) <= column_name_set
        assert set (indi_name_list) <= column_name_set
