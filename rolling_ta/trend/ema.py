from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


class EMA(Indicator):

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
        roll: bool = True,
    ) -> None:
        """Rolling Exponential Moving Average indicator

        https://www.investopedia.com/terms/e/ema.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=14 | Window length.
            weight (np.float64, optional): Default=2.0 | The weight of the EMA's multiplier.
            memory (bool): Default=True | Memory flag, if false removes all information not required for rsi.update().
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation
            roll (bool, optional): Default=True | Calculate remaining indicator values upon instantiation
        """
        super().__init__(data, period, memory, init, roll)
        logger.debug(
            f"EMA init: [data_len={len(data)}, period={period}, memory={memory}]"
        )
        self.set_column_names({"ema": f"ema_{self._period}"})

        self._weight = weight / (period + 1)

        if init:
            self.init()

    def init(self):
        close = self._data["close"]
        count = close.shape[0]
        column = self._column_names["ema"]

        self._window = deque(close[: self._period], maxlen=self._period)
        logger.debug(f"EMA: [window={self._window}]")

        self._window_sum = np.sum(self._window)

        # Initial EMA calculation is SMA
        self._latest_value = self._window_sum / self._period

        if self._memory:
            self._count = count
            self._data[column] = np.nan
            self._data.at[self._period - 1, column] = self._latest_value
        else:
            self._data = None

        if self._roll:
            for i in range(self._period, count):
                self.update(close[i])

                if self._memory:
                    self._data.at[i, column] = self._latest_value

    def update(self, close: np.float64):
        self._close = close
        self.calculate()

        if self._memory:
            self._data.at[self._count, self._column_names["ema"]] = self._latest_value
            self._count += 1

        return self._latest_value

    def calculate(self):
        self._latest_value = (
            (self._close - self._latest_value) * self._weight
        ) + self._latest_value
        return self._latest_value
