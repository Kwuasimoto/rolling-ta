from array import array
from typing import Dict, Literal

import pandas as pd
import numpy as np
from rolling_ta.extras.numba import _typical_price, _vwap
from rolling_ta.indicator import Indicator


class VWAP(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int | Dict[str, int] = 1440,
        memory: bool = True,
        retention: int | None = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        timestamp = self._data["timestamp"].to_numpy(dtype=np.int64)
        volume = self._data["volume"].to_numpy(dtype=np.float64)

        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)
        close = self._data["close"].to_numpy(dtype=np.float64)

        typical_price = np.zeros(close.size, dtype=np.float64)
        _typical_price(high, low, close, typical_price)

        vwap = np.zeros(typical_price.size, dtype=np.float64)
        _vwap(timestamp, typical_price, volume, vwap, self._period_config)

        if self._memory:
            self._vwap = array("d", vwap)

        self.drop_data()
        self.set_initialized()

    def to_array(self, get: Literal["vwap"] = "vwap"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["vwap"] = "vwap",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["vwap"] = "vwap",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
