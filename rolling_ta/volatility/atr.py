from pandas import DataFrame
from rolling_ta.indicator import Indicator

import pandas as pd
import numpy as np


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
        https://www.investopedia.com/terms/a/atr.asp
        https://pypi.org/project/ta/

    Attributes
    ----------
    _atr : pd.Series
        A pandas Series storing the calculated ATR values over the specified period.
    _atr_latest : float
        The most recent ATR value.
    _tr : TrueRange
        An instance of the TrueRange indicator used to calculate the True Range values.
    _period : int
        The number of periods used for the ATR calculation.

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

    _atr: pd.Series
    _atr_latest = np.nan

    def __init__(
        self, data: DataFrame, period: int = 14, memory: bool = True, init: bool = True
    ) -> None:
        super().__init__(data, period, memory, init)

        if self._init:
            self.init()

    def init(self):
        high = self._data["high"]
        low = self._data["low"]
        close = self._data["close"]
        close_p = close.shift(1)

        tr = pd.DataFrame(
            data=[high - low, (high - close_p).abs(), (low - close_p).abs()]
        ).max()

        atr = pd.Series(np.zeros(close.shape[0]))
        atr.iat[self._period - 1] = tr[: self._period].mean()

        self._n_1 = self._period - 1
        for i in range(self._period, close.shape[0]):
            atr.iat[i] = ((atr.iat[i - 1] * self._n_1) + tr[i]) / self._period

        if self._memory:
            self._count = close.shape[0]
            self._atr = atr

        self._close_p = close.iat[-1]
        self._atr_latest = self._atr.iat[-1]

    def update(self, data: pd.Series):
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr = np.max(
            [high - low, np.abs(high - self._close_p), np.abs(low - self._close_p)]
        )

        atr = ((self._atr_latest * self._n_1) + tr) / self._period

        if self._memory:
            self._atr[self._count] = atr
            self._count += 1

        self._close_p = close
        self._atr_latest = atr
        return super().update(data)

    def atr(self):
        return self._atr

    def atr_latest(self):
        return self._atr_latest
