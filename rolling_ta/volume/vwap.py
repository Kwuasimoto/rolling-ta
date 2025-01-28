from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _typical_price, _vwap
from rolling_ta.indicator import Indicator

VWAPKeys = Literal["vwap"]
VWAPPeriods = Literal["vwap"]


class VWAP(Indicator):

    _keys: List[VWAPKeys] = ["vwap"]
    _period_config: Dict[VWAPPeriods, int] = {"vwap": 1440}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[VWAPKeys] = _keys,
        period_config: Dict[VWAPPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
    ) -> None:
        super().__init__(data, keys, period_config, memory, retention, init)
        if self._init:
            self.calc()

    def calc(self):
        timestamp = self._data.index.to_numpy(dtype=np.int64)
        volume = self._data["volume"].to_numpy(dtype=np.float64)

        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)
        close = self._data["close"].to_numpy(dtype=np.float64)

        typical_price = np.zeros(close.size, dtype=np.float64)
        _typical_price(high, low, close, typical_price)

        vwap = np.zeros(typical_price.size, dtype=np.float64)
        _vwap(timestamp, typical_price, volume, vwap, self._period_config["vwap"])

        if self._memory:
            self._vwap = array("d", vwap)

        self.drop_data()
        self.set_initialized()

        return self

    def fit(self, data, period_config: VWAPPeriods = _period_config):
        return super().fit(data, period_config)

    def to_array(self, get: VWAPKeys = "vwap"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: VWAPKeys = "vwap",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: VWAPKeys = "vwap",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
