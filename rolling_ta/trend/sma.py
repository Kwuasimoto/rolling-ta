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

    _sma: pd.Series

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 12,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        """Rolling Simple Moving Average indicator

        https://www.investopedia.com/terms/s/sma.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=12 | Window length.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
        """
        super().__init__(data, period, memory, init)
        logger.debug(
            f"SMA init: [data_len={len(data)}, period={period}, memory={memory}, init={init}]"
        )

        if init:
            self.init()

    def init(self):
        close = self._data["close"]

        self._window = deque(close[-self._period :], maxlen=self._period)
        self._window_sum = np.sum(self._window)

        sma = close.rolling(window=self._period, min_periods=self._period).mean()
        self._latest_sma = sma.iloc[-1]

        # Use memory for sma.
        if self._memory:
            self._count = close.shape[0]
            self._sma = sma

        # Remove dataframe to avoid memory consumption.
        self._data = None

    def update(self, data: pd.Series):
        """Perform rolling update.

        data must be a pd.Series object fetched using .iloc[index | condition] with shape(6,)

        Args:
            data (pd.Series): pd.Series with a "close" column.

        Returns:
            _type_: _description_
        """
        close = data["close"]
        first_close = self._window[0]

        self._window.append(close)
        next_close = self._window[-1]

        self._window_sum = (self._window_sum - first_close) + next_close
        self._latest_sma = self._window_sum / self._period

        if self._memory:
            self._sma[self._count] = self._latest_sma
            self._count += 1

    def sma(self):
        return self._sma

    def latest_sma(self):
        return self._latest_sma
