from array import array
from typing import Deque
import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _rsi
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger

from collections import deque


class NumbaRSI(Indicator):
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
        if init:
            self.init()

    def init(self):
        close = self._data["close"].to_numpy(np.float64)

        rsi, avg_gain, avg_loss, close_p = _rsi(close, self._period_config)

        if self._memory:
            self._rsi = array("f", rsi)

        self.drop_data()

    def rsi(self):
        return pd.Series(self._rsi)


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

        self._alpha = 1 / period_config
        if init:
            self.init()

    def init(self):
        """Calculate the initial RSI values based on historical data."""
        close = self._data["close"]

        delta = close.diff(1)

        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Phase-1 Start (SMA)
        initial_avg_gain = np.mean(gains[: self._period_config])
        initial_avg_loss = np.mean(losses[: self._period_config])

        initial_rsi = (100 * initial_avg_gain) / (initial_avg_gain + initial_avg_loss)

        rsi = pd.Series(index=self._data.index)
        rsi[self._period_config - 1] = initial_rsi
        # Phase-1 End

        # Phase 2 Start (EMA)
        emw_gains = pd.Series(gains, index=close.index)
        emw_losses = pd.Series(losses, index=close.index)

        emw_gains = emw_gains.ewm(
            alpha=self._alpha, min_periods=self._period_config, adjust=False
        ).mean()
        emw_losses = emw_losses.ewm(
            alpha=self._alpha, min_periods=self._period_config, adjust=False
        ).mean()

        emw_rsi = (100 * emw_gains) / (emw_gains + emw_losses)

        rsi[self._period_config :] = emw_rsi[self._period_config :]
        self._rsi_latest = rsi.iloc[-1]

        if self._memory:
            self._rsi = rsi
            self._count = close.shape[0]

            # if self._retention:
            #     self.apply_retention()

        self._prev_price = close.iloc[-1]
        self._emw_gain = emw_gains.iloc[-1]
        self._emw_loss = emw_losses.iloc[-1]

        self.drop_data()

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

            # if self._retention:
            #     self._rsi = self.apply_retention(self._rsi)

    def apply_retention(self):
        self._rsi = self._rsi.tail(self._retention)

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
