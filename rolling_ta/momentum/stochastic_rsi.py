from array import array
from collections import deque
from typing import Dict, Literal, Union

import numpy as np
from rolling_ta.extras.numba import _stoch_k, _stoch_d
from rolling_ta.indicator import Indicator
from rolling_ta.momentum import RSI, RSI
import pandas as pd


class StochasticRSI(Indicator):

    _period_default = {"rsi": 14, "stoch": 10, "k": 3, "d": 3}

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Dict[str, int] = {"rsi": 14, "stoch": 10, "k": 3, "d": 3},
        memory: bool = True,
        retention: Union[int | None] = 20000,
        init: bool = True,
        rsi: Union[RSI | None] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        self._rsi = (
            RSI(data, period_config["rsi"], memory, retention, init)
            if rsi is None
            else rsi
        )

        self._k_period = self.period("stoch")
        self._k_smoothing = self.period("k")
        self._d_smoothing = self.period("d")

        if self._init:
            self.init()

    def init(self, rsi: np.ndarray[np.float64] | None = None):
        if not self._rsi._initialized:
            self._rsi.init()

        rsi = self._rsi.to_numpy()
        stoch_k = np.zeros(rsi.size, dtype=np.float64)

        self._window = _stoch_k(
            rsi,
            stoch_k,
            self._rsi._period_config,
            self._k_period,
            self._k_smoothing,
        )

        if self._d_smoothing > 0:
            stoch_d = np.array(stoch_k, dtype=np.float64)
            _stoch_d(
                stoch_k,
                stoch_d,
                self._rsi._period_config,
                self._k_period,
                self._d_smoothing,
            )

        if self._memory:
            self._k = array("d", stoch_k)

            if stoch_d is not None:
                self._d = array("d", stoch_d)

    def update(self, data: pd.Series):
        return super().update(data, __name__)

    def to_array(self, get: Literal["k", "d"] = "k"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["k", "d"] = "k",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["k", "d"] = "k",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
