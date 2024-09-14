from typing import Deque
from collections import deque

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _sma, _sma_update
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


class SMA(Indicator):
    """
    A class to represent the Simple Moving Average (SMA) indicator.

    The SMA calculates the average of a selected range of prices by the number of periods in that range.
    It smooths out price data to help identify trends over time. This class computes the SMA using historical
    price data over a specified period.

    Material
    --------
        https://www.investopedia.com/terms/s/sma.asp
        https://pypi.org/project/ta/

    Attributes
    ----------
    _sma : pd.Series
        A pandas Series storing the calculated SMA values for each period.
    _sma_latest : float
        The latest SMA value.
    _window : deque
        A deque holding the most recent closing prices within the specified window.
    _window_sum : float
        The sum of values within the window for fast SMA calculation.

    Methods
    -------
    **__init__(data: pd.DataFrame, period: int = 12, memory: bool = True, init: bool = True)** -> None

        Initializes the SMA indicator with the given data, period, and options.

    **init()** -> None

        Calculates the initial SMA values based on the provided data.

    **update(data: pd.Series)** -> None

        Updates the SMA based on new incoming data.

    **sma()** -> pd.Series

        Returns the stored SMA values if memory is enabled.

    **sma_latest()** -> float

        Returns the latest SMA value.
    """

    _sma: pd.Series
    _sma_latest = np.nan

    _window: Deque[np.float64]
    _window_sum = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = 12,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        """
        Initializes the SMA indicator with the given data, period, and options.

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=12 | Window length.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
        """
        super().__init__(data, period_config, memory, retention, init)
        # logger.debug(
        #     f"SMA init: [data_len={len(data)}, period={period_config}, memory={memory}, init={init}]"
        # )

        if init:
            self.init()

    def init(self):
        """
        Calculates the initial SMA values based on the provided data.

        This method computes the initial SMA values using the specified window length. It also initializes
        internal attributes required for rolling updates and stores the computed SMA values if memory is enabled.
        """
        close = self._data["close"]

        sma = (
            close.rolling(window=self._period_config, min_periods=self._period_config)
            .mean()
            .fillna(0)
        )

        self._sma_latest = sma.iloc[-1]

        # Use memory for sma.
        if self._memory:
            self._count = close.shape[0]
            self._sma = sma

        self._window = deque(close[-self._period_config :], maxlen=self._period_config)
        self._window_sum = np.sum(self._window)

        # Remove dataframe to avoid memory consumption.
        self.drop_data()

    def update(self, data: pd.Series):
        """
        Updates the SMA based on new incoming data.

        Args:
            data (pd.Series): A pandas Series containing the new data. Must include a "close" value.
        """
        close = data["close"]

        first_close = self._window[0]
        self._window.append(close)

        self._window_sum = (self._window_sum - first_close) + close
        self._sma_latest = self._window_sum / self._period_config

        if self._memory:
            self._sma[self._count] = self._sma_latest
            self._count += 1

        # if self._retention:
        #     self._sma = self.apply_retention(self._sma)

    def sma(self):
        """
        Returns the stored SMA values if memory is enabled.

        Returns:
            pd.Series: A pandas Series containing the SMA values.

        Raises:
            MemoryError: If function called and memory = False
        """
        if not self._memory:
            raise MemoryError("SMA._memory = False")
        return self._sma

    def sma_latest(self):
        """
        Returns the latest SMA value.

        Returns:
            float: The most recent SMA value.
        """
        return self._sma_latest


# Numba enhanced SMA (init: 5-10x faster, update 4-5x faster)
class NumbaSMA(Indicator):
    _sma: np.ndarray[np.float64]
    _sma_latest = np.nan

    _window: Deque[np.float64]
    _window_sum = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = 12,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        if init:
            self.init()

    def init(self):
        close = self._data["close"].to_numpy(dtype=np.float64)
        sma, current_sum = _sma(close, self._period_config)

        self._window = deque(close[-self._period_config :], maxlen=self._period_config)
        self._window_sum = current_sum

        self._sma_latest = sma[-1]

        if self._memory:
            self._sma = sma

        self.drop_data()

    def update(self, data: pd.Series):
        close = data["close"]
        close_f = self._window[0]

        sma_latest, current_sum = _sma_update(
            self._window_sum, close, close_f, self._period_config
        )

        self._sma_latest = sma_latest
        self._window_sum = current_sum

        self._window.append(close)

        if self._memory:
            self._sma = np.append(self._sma, self._sma_latest)

    def sma(self):
        if not self._memory:
            raise MemoryError("SMA._memory = False")
        return pd.Series(self._sma)

    def sma_latest(self):
        return self._sma_latest
