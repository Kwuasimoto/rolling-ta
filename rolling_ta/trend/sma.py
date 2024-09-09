from typing import Deque
from collections import deque

import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


# Math derived from chatGPT + https://www.investopedia.com/terms/s/sma.asp
class SMA(Indicator):

    _sma = np.nan
    _window: Deque[np.float64]
    _window_sum = np.nan

    def __init__(
        self, data: pd.DataFrame, period: int = 12, memory=True, init=True
    ) -> None:
        super().__init__(data, period, memory, init)
        logger.debug(
            f"Building SMA: [data_len={len(data)}, period={period}, memory={memory}]"
        )

        if init:
            self.init()

    def init(self):
        close = self._data["close"]
        self._window = deque(close[: self._period], maxlen=self._period)
        logger.debug(f"SMA: [window={self._window}]")

        # Use a deque to maintain O(1) efficiency
        self._window_sum = np.sum(self._window)

        logger.debug(f"SMA: [sum={self._window_sum}]")

        # Calculate initial SMA
        self._sma = self.calculate()

        # Perform memory optimization.
        if self._memory:
            self._count = len(close)
            self._data[f"sma_{self._period}"] = np.nan
            self._data.at[self._period - 1, f"sma_{self._period}"] = self._sma
        else:
            self._data = None

        # Calculate the rest of the SMA
        for i in range(self._period, len(close)):
            self.update(close[i])

            if self._memory:
                self._data.at[i, f"sma_{self._period}"] = self._sma

    def update(self, close: np.float64):
        # Reduce sum by first value in the deque.
        self._window_sum -= self._window[0]
        self._window_sum += close

        self._window.append(close)
        self._sma = self.calculate()

        if self._memory:
            self._data.at[self._count, f"sma_{self._period}"] = self._sma
            self._count += 1

    def calculate(self):
        self._sma = self._window_sum / self._period
        return self._sma

    def data(self):
        return self._data[f"sma_{self._period}"]
