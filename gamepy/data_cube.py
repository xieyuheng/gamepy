import numpy as np
import pandas as pd

def list_intersect (x, y):
    return list (filter (lambda e: e in y, x))

class data_cube_c:
    def __init__ (self, df, attr_name_list, indi_name_list):
        column_name_list = list (df.columns)
        attr_name_list = list_intersect (attr_name_list, column_name_list)
        indi_name_list = list_intersect (indi_name_list, column_name_list)
        name_list = attr_name_list + indi_name_list
        df = pd.DataFrame (df, columns = name_list)
        self.df = df
        self.attr_name_list = attr_name_list
        self.indi_name_list = indi_name_list

    def __repr__ (self):
        s = "\n"
        s += self.df.__repr__ ()
        s += "\n"
        s += "attr_name_list = {}\n" .format (self.attr_name_list)
        s += "indi_name_list = {}\n" .format (self.indi_name_list)
        return s

    def proj (self, itemdict):
        df = self.df
        for k, v in itemdict.items ():
            df = df [ df [k] .isin ([v]) ]
        return data_cube_c (
            df,
            self.attr_name_list,
            self.indi_name_list)

    def proj_drop (self, itemdict):
        cube = self.proj (itemdict)
        return data_cube_c (
            cube.df.drop (columns = itemdict.keys ()),
            self.attr_name_list,
            self.indi_name_list)

    def retract (self, attr_name_list):
        return
