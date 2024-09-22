from array import array
from typing import Deque
import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _rsi, _rsi_update
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger

from collections import deque


class RSI(Indicator):
    """
    Relative Strength Index (RSI) indicator.

    The RSI is a momentum oscillator that measures the speed and change of price
    movements. It oscillates between 0 and 100 and is used to identify overbought
    or oversold conditions in an asset. This class calculates the RSI using
    historical price data over a specified period.

    Material
    --------
        https://www.investopedia.com/terms/r/rsi.asp
    """

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        """
        Initialize the RSI indicator.

        Args:
            data (pd.DataFrame): The initial dataframe containing price data with a 'close' column.
            period (int): Default=14 | The period over which to calculate the RSI.
            memory (bool): Default=True | Whether to store RSI values in memory.
            retention (int): Default=20000 | The maximum number of RSI values to store in memory
            init (bool): Default=True | Whether to calculate the initial RSI values upon instantiation.
        """
        super().__init__(data, period_config, memory, retention, init)
        self.alpha = 1 / self._period_config
        if init:
            self.init()

    def init(self):
        close = self._data["close"].to_numpy(np.float64)
        rsi = np.zeros(close.size, dtype=np.float64)
        gains = np.zeros(close.size, dtype=np.float64)
        losses = np.zeros(close.size, dtype=np.float64)

        rsi, avg_gain, avg_loss, close_p = _rsi(
            close,
            rsi,
            gains,
            losses,
            self._period_config,
        )

        if self._memory:
            self._rsi = array("f", rsi)

        self._avg_gain = avg_gain
        self._avg_loss = avg_loss
        self._close_p = close_p

        self.drop_data()

    def update(self, data: pd.Series):
        close = data["close"]

        rsi, avg_gain, avg_loss = _rsi_update(
            close,
            self._close_p,
            self._avg_gain,
            self._avg_loss,
            self.alpha,
        )

        if self._memory:
            self._rsi.append(rsi)

        self._avg_gain = avg_gain
        self._avg_loss = avg_loss
        self._close_p = close

    def rsi(self):
        return pd.Series(self._rsi)
