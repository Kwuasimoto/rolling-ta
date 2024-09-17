from typing import Optional, Type, Union
from numba import types, typed, typeof
from numba.experimental import jitclass
import numpy as np


## If we want to hyper optimize further in the future:
# - Use composition with this class instead of extending Indicator on NumbaIndicators.
# - jitclass type hints must use Numba type instances (ex: types.boolean, types.string)
#       Not Scalars such as types.Boolean
@jitclass(
    [
        ("_init", types.boolean),
        ("_memory", types.boolean),
        ("_data", types.DictType(types.string, types.ListType(types.f8))),
    ]
)
class NumbaIndicator:
    def __init__(
        self,
        data: types.DictType,
        init: types.Boolean = True,
        memory: types.Boolean = False,
    ) -> None:
        self._data = data
        self._init = init
        self._memory = memory

    def drop_data(self):
        d = typed.Dict()
        d["_removed"] = typed.List([np.nan])
        self._data = d
