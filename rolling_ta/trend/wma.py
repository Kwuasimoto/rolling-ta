from array import array
from typing import Dict, Literal, Union

import numpy as np
import pandas as pd
from rolling_ta.extras.numba import _wma
from rolling_ta.indicator import Indicator


class WMA(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int | Dict[str, int] = 14,
        memory: bool = True,
        retention: int | None = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        close = self._data["close"].to_numpy(dtype=np.float64)

        wma = np.zeros(close.size, dtype=np.float64)

        _wma(close, wma, self._period_config)

        if self._memory:
            self._wma = array("d", wma)

        self.drop_data()
        self.set_initialized()

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
