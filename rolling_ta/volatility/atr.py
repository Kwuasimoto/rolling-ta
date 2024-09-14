from pandas import DataFrame
from rolling_ta.indicator import Indicator
from rolling_ta.extras.numba import _atr, _atr_update
from rolling_ta.volatility import TrueRange


import pandas as pd
import numpy as np

from typing import Dict, Optional


class AverageTrueRange(Indicator):
    """
    Rolling Average True Range (ATR) indicator.

    The Average True Range (ATR) is a technical analysis indicator that measures market volatility.
    It is derived from the True Range (TR), which takes the greatest of the following:
    - The current high minus the current low.
    - The absolute value of the current high minus the previous close.
    - The absolute value of the current low minus the previous close.

    The ATR is calculated as an exponentially smoothed moving average of the True Range over a specified period.

    Material
    --------
    - https://www.investopedia.com/terms/a/atr.asp
    - https://pypi.org/project/ta/

    Attributes
    ----------
    _atr : pd.Series
        A pandas Series storing the calculated ATR values over the specified period.
    _atr_latest : float
        The most recent ATR value.

    Methods
    -------
    **__init__(data: pd.DataFrame, period: int = 14, memory: bool = True, init: bool = True)** -> None

        Initializes the ATR indicator with the given data, period, and options.

    **init()** -> None

        Calculates the initial ATR values using the True Range over the specified period.

    **update(data: pd.Series)** -> None

        Updates the ATR based on new incoming data (high, low, close).

    **atr()** -> pd.Series

        Returns the stored ATR values if memory is enabled.

    **atr_latest()** -> float

        Returns the latest ATR value.
    """

    _tr: TrueRange

    _atr: pd.Series
    _atr_latest = np.nan

    def __init__(
        self,
        data: DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
        true_range: Optional[TrueRange] = None,
    ) -> None:
        """
        Initialize the ATR indicator.

        Args:
            data (pd.DataFrame): The initial dataframe containing price data with 'high', 'low', 'close' columns.
            period (int): Default=14 | The period over which to calculate the ATR.
            memory (bool): Default=True | Whether to store ATR values in memory.
            init (bool): Default=True | Whether to calculate the initial ATR values upon instantiation.
        """
        super().__init__(data, period_config, memory, retention, init)
        self._tr = (
            TrueRange(data, period_config, memory, retention, init)
            if true_range is None
            else true_range
        )

        if self._init:
            self.init()

    def init(self):
        """Calculate the initial ATR values based on historical data."""

        # Check if True Range was initialized on instantiation.
        if not self._tr._init:
            self._tr.init()

        close = self._data["close"]

        tr = self._tr.tr()

        atr = pd.Series(np.zeros(close.shape[0]))
        atr.iat[self._period_config - 1] = tr[: self._period_config].mean()

        self._n_1 = self._period_config - 1
        for i in range(self._period_config, close.shape[0]):
            atr.iat[i] = ((atr.iat[i - 1] * self._n_1) + tr[i]) / self._period_config

        if self._memory:
            self._count = close.shape[0]
            self._atr = atr

        self._atr_latest = self._atr.iat[-1]

        self.drop_data()

    def update(self, data: pd.Series):
        """
        Update the ATR with new price data.

        Args:
            data (pd.Series): The latest data containing 'high', 'low', 'close' prices.
        """
        self._tr.update(data)

        atr = (
            (self._atr_latest * self._n_1) + self._tr._tr_latest
        ) / self._period_config

        if self._memory:
            self._atr[self._count] = atr
            self._count += 1

        self._atr_latest = atr

    def atr(self):
        """
        Return the stored ATR values.

        Returns:
            pd.Series: The ATR values calculated over the historical data if memory is enabled.

        Raises:
            MemoryError: if function called and memory = False
        """
        if not self._memory:
            raise MemoryError("ATR._memory = False")
        return self._atr

    def atr_latest(self):
        """
        Return the most recent RSI value.

        Returns:
            float: The latest RSI value.
        """
        return self._atr_latest


class NumbaAverageTrueRange(Indicator):

    def __init__(
        self,
        data: DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
        true_range: Optional[TrueRange] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        self._tr = TrueRange(data, period_config) if true_range is None else true_range
        self._n_1 = self._period_config - 1
        if self._init:
            self.init()

    def init(self):
        if not self._init:
            self._tr.init()

        atr = _atr(
            self._tr.tr().to_numpy(np.float64),
            self._period_config,
        )

        if self._memory:
            self._atr = atr

        self._atr_latest = self._atr[-1]

        self.drop_data()

    def update(self, data: pd.Series):

        self._tr.update(data)

        self._atr_latest = _atr_update(
            self._atr_latest,
            self._tr._tr_latest,
            self._period_config,
            self._n_1,
        )

        if self._memory:
            self._atr = np.append(self._atr, self._atr_latest)

    def atr(self):
        return self._atr
