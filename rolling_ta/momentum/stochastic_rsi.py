from array import array
from collections import deque
from typing import Dict, Union

import numpy as np
from rolling_ta.extras.numba import _stoch_k, _stoch_rsi
from rolling_ta.indicator import Indicator
from rolling_ta.momentum import RSI, RSI
import pandas as pd


class StochasticRSI(Indicator):

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

        self._stoch_period = self.period("stoch")
        self._k_period = self.period("k")
        self._d_period = self.period("d")

        if self._init:
            self.init()

    def init(self, rsi: np.ndarray[np.float64] = None):
        if not self._init:
            self._rsi.init()

        # rsi = rsi if rsi is not None else self._rsi.rsi().to_numpy(dtype=np.float64)
        rsi = self._rsi.rsi().to_numpy(np.float64)
        window = np.empty(self._stoch_period, dtype=np.float64)
        stoch_rsi = np.zeros(rsi.size, dtype=np.float64)
        # sl = slice(0, 13)
        # dummy_calc = (rsi[13] - min(rsi[0:14])) / (max(rsi[0:14]) - min(rsi[0:14]))

        for i in range(self._rsi._period_config, rsi.size - self._stoch_period):
            y = i + self._stoch_period - 1

            # Use the rsi at rsi_period(14) + stoch_period(10) = 24
            curr_rsi = rsi[y]

            # Use rsi values from rsi_period(14) to 24
            max_rsi = max(rsi[i:y])
            min_rsi = min(rsi[i:y])
            stoch_rsi[y] = (curr_rsi - min_rsi) / (max_rsi - min_rsi)

        # _stoch_rsi(rsi, window, stoch_rsi, self._rsi._period_config, self._stoch_period)

        # k = np.zeros(rsi.size, dtype=np.float64)
        # _stoch_k(stoch_rsi, k, self._k_period)

        if self._memory:
            self._stoch_rsi = array("d", stoch_rsi)
            self._k = array("d", stoch_rsi)

    def update(self, data: pd.Series):
        return super().update(data)

    def stoch_rsi(self):
        return pd.Series(self._stoch_rsi)
