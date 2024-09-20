from array import array
from collections import deque
from typing import Dict, Union

import numpy as np
from rolling_ta.extras.numba import _stoch_k, _stoch_rsi
from rolling_ta.indicator import Indicator
from rolling_ta.momentum import RSI, NumbaRSI
import pandas as pd


class NumbaStochasticRSI(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Dict[str, int] = {"rsi": 14, "stoch": 10, "k": 3, "d": 3},
        memory: bool = True,
        retention: Union[int | None] = 20000,
        init: bool = True,
        rsi: Union[NumbaRSI | None] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        self._rsi = (
            NumbaRSI(data, period_config["rsi"], memory, retention, init)
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

    def init(self):
        if not self._init:
            self._rsi.init()

        rsi = self._rsi.rsi()

        rsi_mins = rsi.rolling(self._stoch_period, min_periods=self._stoch_period).min()
        rsi_maxs = rsi.rolling(self._stoch_period, min_periods=self._stoch_period).max()

        stoch_rsi = (rsi - rsi_mins) / (rsi_maxs - rsi_mins)

        k = stoch_rsi.rolling(self._k_period, min_periods=self._k_period).mean()
        d = k.rolling(self._d_period, min_periods=self._d_period).mean()

        if self._memory:
            self._stoch_rsi = stoch_rsi
            self._k = k
            self._d = d
            self._count = rsi.shape[0]

        self._stoch_rsi_latest = self._stoch_rsi.iat[-1]
        self._k_latest = self._k.iat[-1]
        self._d_latest = self._d.iat[-1]

        # Required for k calc if mem is off
        self._stoch_deque = deque(
            stoch_rsi.iloc[-self._k_period :], maxlen=self._k_period
        )
        # Required for d calc if mem is off
        self._k_deque = deque(k.iloc[-self._d_period :], maxlen=self._d_period)

        self.drop_data()

    def update(self, data: pd.Series):

        self._rsi.update(data)

        rsi = self._rsi.rsi().iloc[-self._stoch_period :]

        rsi_min = np.min(rsi)
        rsi_max = np.max(rsi)

        self._stoch_rsi_latest = (rsi.iat[-1] - rsi_min) / (rsi_max - rsi_min)

        self._stoch_deque.append(self._stoch_rsi_latest)
        k = np.mean(self._stoch_deque)

        self._k_deque.append(k)
        d = np.mean(self._k_deque)

        if self._memory:
            self._stoch_rsi[self._count] = self._stoch_rsi_latest
            self._k[self._count] = k
            self._d[self._count] = d
            self._count += 1

    def rsi(self):
        return self._rsi.rsi()

    def stoch_rsi(self):
        return self._stoch_rsi

    def k(self):
        return self._k

    def d(self):
        return self._d
