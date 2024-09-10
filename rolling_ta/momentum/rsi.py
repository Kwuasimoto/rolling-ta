from typing import Deque
import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger

from collections import deque


# Math derived from chatGPT + https://www.investopedia.com/terms/r/rsi.asp
class RSI(Indicator):
    """Rolling RSI Indicator https://www.investopedia.com/terms/r/rsi.asp"""

    _prev_price = np.nan

    _alpha = np.nan
    _gains: Deque
    _losses: Deque
    _emw_gain = np.nan
    _emw_loss = np.nan

    _rsi: pd.Series
    _latest_rsi = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        """Rolling RSI indicator

        https://www.investopedia.com/terms/r/rsi.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=14 | Window length.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
            roll (bool, optional): Default=True | Calculate remaining indicator values upon instantiation.
        """
        super().__init__(data, period, memory, init)
        self._alpha = 1 / period
        if init:
            self.init()

    def init(self):
        close = self._data["close"]
        # Store the prev price for subsequent RSI updates.

        delta = close.diff(1)

        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Phase-1 Start (SMA)
        initial_avg_gain = np.mean(gains[: self._period])
        initial_avg_loss = np.mean(losses[: self._period])

        # initial_rs = (
        #     initial_avg_gains / initial_avg_losses
        #     if initial_avg_losses != 0
        #     else np.inf
        # )
        # initial_rsi = 100 - (100 / (1 + initial_rs)) if initial_avg_losses != 0 else 100
        # (100 * emw_gains) / (emw_gains + emw_losses)
        initial_rsi = (100 * initial_avg_gain) / (initial_avg_gain + initial_avg_loss)

        rsi = pd.Series(index=self._data.index)
        rsi[self._period - 1] = initial_rsi
        # Phase-1 End

        # Phase 2 Start (EMA)
        emw_gains = pd.Series(gains, index=close.index)
        emw_losses = pd.Series(losses, index=close.index)

        emw_gains = emw_gains.ewm(
            alpha=self._alpha, min_periods=self._period, adjust=False
        ).mean()
        emw_losses = emw_losses.ewm(
            alpha=self._alpha, min_periods=self._period, adjust=False
        ).mean()

        emw_rsi = (100 * emw_gains) / (emw_gains + emw_losses)
        # emw_rsi = 100 - (100 / (1 + (emw_gains / emw_losses)))
        rsi[self._period :] = emw_rsi[self._period :]
        self._latest_rsi = rsi.iloc[-1]

        if self._memory:
            self._rsi = rsi
            self._count = close.shape[0]

        self._data = None

        # Store information required for rolling updates.
        self._prev_price = close.iloc[-1]
        self._gains = deque(gains[-self._period :], maxlen=self._period)
        self._losses = deque(losses[-self._period :], maxlen=self._period)

        self._emw_gain = emw_gains.iloc[-1]
        self._emw_loss = emw_losses.iloc[-1]

    def update(self, data: pd.Series):
        # Get the delta in price, and calculate gain/loss
        close = data["close"]
        delta = close - self._prev_price
        self._prev_price = close

        gain = max(delta, 0)
        loss = -min(delta, 0)

        self._gains.append(gain)
        self._losses.append(loss)

        # FORMULA: emwa_new = alpha * value + (1 - alpha) * ewma_prev
        # REFACTOR (-1 mul): alpha * (value - ewma_prev) + ewma_prev

        # self._emw_gain = self._alpha * gain + (1 - self._alpha) * self._emw_gain
        # self._emw_loss = self._alpha * loss + (1 - self._alpha) * self._emw_loss
        self._emw_gain = self._alpha * (gain - self._emw_gain) + self._emw_gain
        self._emw_loss = self._alpha * (loss - self._emw_loss) + self._emw_loss

        # FORMULA: 100 - (100 / (1 + a / b))
        # REFACTOR (+1 mul, -1 sub, -1 div): (100 * a) / (a + b)

        # self._rsi[self._count] = 100 - (100 / (1 + self._emw_gain / self._emw_loss))
        self._latest_rsi = (100 * self._emw_gain) / (self._emw_gain + self._emw_loss)

        if self._memory:
            self._rsi[self._count] = self._latest_rsi
            self._count += 1

    def rsi(self):
        return self._rsi

    def latest_rsi(self):
        return self._latest_rsi
