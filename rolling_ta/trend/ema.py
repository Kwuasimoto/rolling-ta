from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


class EMA(Indicator):

    _ema = np.nan
    _close = np.nan

    _window: Deque[np.float64]
    _window_sum = np.nan
    _weight = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        weight: np.float64 = 2.0,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        super().__init__(data, period, memory, init)
        logger.debug(
            f"Building SMA: [data_len={len(data)}, period={period}, memory={memory}]"
        )

        self._weight = weight / (period + 1)

        if init:
            self.init()

    def init(self):
        close = self._data["close"]
        self._window = deque(close[: self._period], maxlen=self._period)
        logger.debug(f"EMA: [window={self._window}]")

        self._window_sum = np.sum(self._window)

        # Initial EMA calculation is SMA
        self._ema = self._window_sum / self._period

        if self._memory:
            self._count = len(close)
            self._data[f"ema_{self._period}"] = np.nan
            self._data.at[self._period - 1, f"ema_{self._period}"] = self._ema
        else:
            self._data = None

        for i in range(self._period, len(close)):
            self.update(close[i])

            if self._memory:
                self._data.at[i, f"ema_{self._period}"] = self._ema
                self._count += 1

    def update(self, close: np.float64):
        self._close = close
        self.calculate()

    def calculate(self):
        self._ema = ((self._close - self._ema) * self._weight) + self._ema
        return self._ema

    def data(self):
        if not self._memory:
            raise MemoryError("Ema memory is false, please enable memory or use ")
        return self._data[f"ema_{self._period}"]
