from array import array
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _wma
from rolling_ta.indicator import Indicator


class WMA(Indicator):

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: int = 14,
        memory: bool = True,
        retention: Optional[int] = None,
        columns: Optional[list[str]] = None,
        init: bool = False,
    ) -> None:
        super().__init__(data, period_config, memory, retention, columns, init)

        if self._init:
            self.set_columns(columns)
            self.calc()

    def calc(self):
        close = self._data["close"].to_numpy(dtype=np.float64)

        wma = np.zeros(close.size, dtype=np.float64)

        _wma(close, wma, self._period_config)

        if self._memory:
            self._wma = array("d", wma)

        if self._columns is None:
            self.set_columns()

        self.drop_data()
        self.set_initialized()

        return self

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            f"wma_{self._period_config}" if columns is None else columns,
            name,
        )

    def to_array(self, get: Literal["wma"] = "wma"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["wma"] = "wma",
        dtype: Union[np.dtype, None] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["wma"] = "wma",
        dtype: Union[type, None] = float,
        name: Union[str, None] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
