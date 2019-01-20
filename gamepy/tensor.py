import numpy as np
import pandas as pd

class tensor_c:
    def __init__ (self, axes, ndarray):
        self.axes  = pd.Series (axes)
        self.shape = tuple (map (len, self.axes.values))
        assert self.shape == ndarray.shape
        self.ndarray = ndarray

    # idx: pd.Series
    #   index: axis_name
    #   value: axis_var

    def idx_c (self, idx):
        return pd.Series (idx, index = self.axes.index)

    # idx_item: (axis_name, axis_var)

    def idx_item_to_index (self, idx_item):
        axis_name, axis_var = idx_item
        if pd.isna (axis_var):
            return slice (None)
        else:
            # [todo]
            # we can improved `.index`
            #   if `self.axes` is pd.Series
            return self.axes [axis_name] .index (axis_var)

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
