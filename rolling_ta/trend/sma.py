from typing import Deque
from collections import deque

import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


# Math derived from chatGPT + https://www.investopedia.com/terms/s/sma.asp
class SMA(Indicator):

    _window: Deque[np.float64]
    _window_sum = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 12,
        memory: bool = True,
        init: bool = True,
        roll: bool = True,
    ) -> None:
        """Rolling Simple Moving Average indicator

        https://www.investopedia.com/terms/s/sma.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=12 | Window length.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
            roll (bool, optional): Default=True | Calculate remaining indicator values upon instantiation.
        """
        super().__init__(data, period, memory, init, roll)
        logger.debug(
            f"SMA init: [data_len={len(data)}, period={period}, memory={memory}, init={init}]"
        )

        self.set_column_names({"sma": f"sma_{self._period}"})

        if init:
            self.init()

    def init(self):
        close = self._data["close"]
        count = close.shape[0]
        column = self._column_names["sma"]

        self._window = deque(close[: self._period], maxlen=self._period)
        logger.debug(f"SMA: [window={self._window}]")

        # Use a deque to maintain O(1) efficiency
        self._window_sum = np.sum(self._window)

        logger.debug(f"SMA: [sum={self._window_sum}]")

        # Calculate initial SMA
        self._latest_value = self.calculate()

        # Perform memory optimization.
        if self._memory:
            self._count = count
            self._data[column] = np.nan
            self._data.at[self._period - 1, column] = self._latest_value
        else:
            self._data = None

        # Roll the rest of the SMA
        if self._roll:
            for i in range(self._period, count):
                self.update(close[i])

                if self._memory:
                    self._data.at[i, column] = self._latest_value

    def update(self, close: np.float64):
        # Reduce sum by first value in the deque.
        self._window_sum -= self._window[0]
        self._window_sum += close

        self._window.append(close)
        self._latest_value = self.calculate()

        if self._memory:
            self._data.at[self._count, self._column_names["sma"]] = self._latest_value
            self._count += 1

        return self._latest_value

    def calculate(self):
        self._latest_value = self._window_sum / self._period
        return self._latest_value
