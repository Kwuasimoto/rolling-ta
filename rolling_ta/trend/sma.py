from array import array
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _sma, _sma_update
from rolling_ta.indicator import Indicator


SMAKeys = Literal["sma"]
SMAPeriods = Literal["sma"]


class SMA(Indicator):
    """
    A class to represent the Simple Moving Average (SMA) indicator.

    The SMA calculates the average of a selected range of prices by the number of periods in that range.
    It smooths out price data to help identify trends over time. This class computes the SMA using historical
    price data over a specified period.

    Material
    --------
        https://www.investopedia.com/terms/s/sma.asp
        https://pypi.org/project/ta/
    """

    _keys: List[SMAKeys] = ["sma"]
    _period_config: Dict[SMAPeriods, int] = {"sma": 14}

    _sma_latest = np.nan

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[SMAKeys] = _keys,
        period_config: Dict[SMAPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
    ) -> None:
        super().__init__(
            data,
            keys=keys,
            period_config=period_config,
            memory=memory,
            retention=retention,
            init=init,
        )
        if init:
            self.calc()

    def calc(self):
        close = self._data["close"].to_numpy(dtype=np.float64)
        sma = np.zeros(close.size)

        sma, window, window_sum, latest = _sma(
            close,
            sma,
            self._period_config["sma"],
        )

        self._window = window
        self._window_sum = window_sum
        self._sma_latest = latest

        if self._memory:
            self._sma = array("f", sma)

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):

        latest, window, window_sum = _sma_update(
            data["close"],
            self._window_sum,
            self._window,
            self._period_config["sma"],
        )

        self._sma_latest = latest
        self._window_sum = window_sum
        self._window = window

        if self._memory:
            self._sma.append(latest)

        return self

    def fit(self, data, period_config: SMAPeriods = _period_config):
        return super().fit(data, period_config)

    def to_array(self, get: SMAKeys = "sma"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: SMAKeys = "sma",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: SMAKeys = "sma",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
