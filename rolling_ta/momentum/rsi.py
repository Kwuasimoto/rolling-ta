from typing import Deque
import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger

from collections import deque


# Math derived from chatGPT + https://www.investopedia.com/terms/r/rsi.asp
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
        https://pypi.org/project/ta/

    Attributes
    ----------
    _prev_price : float
        The previous closing price used to calculate the price change.
    _alpha : float
        The smoothing factor for exponential moving averages (EMA).
    _emw_gain : float
        The exponentially weighted moving average of gains.
    _emw_loss : float
        The exponentially weighted moving average of losses.
    _rsi : pd.Series
        A pandas Series storing the calculated RSI values for each period.
    _rsi_latest : float
        The latest RSI value.

    Methods
    -------
    **__init__(data: pd.DataFrame, period: int = 14, memory: bool = True, init: bool = True)** -> None

        Initializes the RSI indicator with the given data, period, and options.

    **init()** -> None

        Calculates the initial RSI values based on the provided data.

    **update(data: pd.Series)** -> None

        Updates the RSI based on new incoming data.

    **rsi()** -> pd.Series

        Returns the stored RSI values if memory is enabled.

    **rsi_latest()** -> float

        Returns the latest RSI value.
    """

    _prev_price = np.nan

    _alpha = np.nan

    _emw_gain = np.nan
    _emw_loss = np.nan

    _rsi: pd.Series
    _rsi_latest = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        """
        Initialize the RSI indicator.

        Args:
            data (pd.DataFrame): The initial dataframe containing price data with a 'close' column.
            period (int): Default=14 | The period over which to calculate the RSI.
            memory (bool): Default=True | Whether to store RSI values in memory.
            init (bool): Default=True | Whether to calculate the initial RSI values upon instantiation.
        """
        super().__init__(data, period, memory, init)
        self._alpha = 1 / period
        if init:
            self.init()

    def init(self):
        """
        Calculate the initial RSI values based on historical data.

        Args:
            None
        """
        close = self._data["close"]

        delta = close.diff(1)

        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Phase-1 Start (SMA)
        initial_avg_gain = np.mean(gains[: self._period])
        initial_avg_loss = np.mean(losses[: self._period])

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
        rsi[self._period :] = emw_rsi[self._period :]
        self._rsi_latest = rsi.iloc[-1]

        if self._memory:
            self._rsi = rsi
            self._count = close.shape[0]

        self._data = None

        self._prev_price = close.iloc[-1]
        self._emw_gain = emw_gains.iloc[-1]
        self._emw_loss = emw_losses.iloc[-1]

    def update(self, data: pd.Series):
        """
        Update the RSI with new price data.

        Args:
            data (pd.Series): The latest data containing a 'close' price.
        """
        close = data["close"]
        delta = close - self._prev_price
        self._prev_price = close

        gain = max(delta, 0)
        loss = -min(delta, 0)

        self._emw_gain = self._alpha * (gain - self._emw_gain) + self._emw_gain
        self._emw_loss = self._alpha * (loss - self._emw_loss) + self._emw_loss

        self._rsi_latest = (100 * self._emw_gain) / (self._emw_gain + self._emw_loss)

        if self._memory:
            self._rsi[self._count] = self._rsi_latest
            self._count += 1

    def rsi(self):
        """
        Return the stored RSI values.

        Returns:
            pd.Series: The RSI values calculated over the historical data if memory is enabled.

        Raises:
            MemoryError: if function called and memory = False
        """
        if not self._memory:
            raise MemoryError("RSI._memory = False")
        return self._rsi

    def rsi_latest(self):
        """
        Return the most recent RSI value.

        Returns:
            float: The latest RSI value.
        """
        return self._rsi_latest
