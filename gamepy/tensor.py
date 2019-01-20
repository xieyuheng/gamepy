import numpy as np
import pandas as pd

class tensor_error_c (Exception):
    pass

# axes: pd.Series <axis_name, pd.Series <axis_var, index>>

def axis_vars_c (axis_vars, axis_name):
    if type (axis_vars) == list:
        return pd.Series (
            range (len (axis_vars)),
            index = axis_vars,
            name = axis_name)
    elif type (axis_vars) == dict:
        return pd.Series (
            axis_vars,
            name = axis_name)
    elif type (axis_vars) == pd.Series:
        return axis_vars
    else:
        raise tensor_error_c

def axes_c (axes):
    axes = pd.Series (axes)
    values = []
    for axis_name, axis_vars in axes.iteritems ():
        values.append (axis_vars_c (axis_vars, axis_name))

    return pd.Series (
        values,
        index = axes.index)

class tensor_c:
    def __init__ (self, axes, ndarray):
        self.axes = axes_c (axes)
        self.shape = tuple (map (len, self.axes.values))
        assert self.shape == ndarray.shape
        self.ndarray = ndarray

    # idx: pd.Series <axis_name, axis_var>

    def idx_c (self, idx):
        return pd.Series (idx, index = self.axes.index)

    # idx_item: (axis_name, axis_var)

    def idx_item_to_index (self, idx_item):
        axis_name, axis_var = idx_item
        if pd.isna (axis_var):
            return slice (None)
        else:
            return self.axes [axis_name] [axis_var]

    def idx_to_index (self, idx):
        return tuple (list (map (
            self.idx_item_to_index,
            idx.items ())))

    def idx (self, idx):
        # -> np.ndarray
        idx = self.idx_c (idx)
        index = self.idx_to_index (idx)
        return self.ndarray [index]

    def proj (self, idx):
        # -> tensor_c
        idx = self.idx_c (idx)
        axes = self.axes [ idx.isna () ]
        ndarray = self.idx (idx)
        return tensor_c (axes, ndarray)
