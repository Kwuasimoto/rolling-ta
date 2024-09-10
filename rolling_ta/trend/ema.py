from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) Indicator.

    The EMA gives more weight to recent prices, making it more responsive to new information compared to the Simple Moving Average (SMA).
    This indicator is commonly used to identify trends and smooth out price data.

    Material
    --------
        https://www.investopedia.com/terms/e/ema.asp

    Attributes
    ----------
    _close : float
        The most recent closing price used for calculating the EMA.
    _ema : pd.Series
        A pandas Series storing the calculated EMA values for each period.
    _ema_latest : float
        The latest EMA value.
    _weight : float
        The weight for the EMA calculation, derived from the period.

    Methods
    -------
    **__init__(data: pd.DataFrame, period: int = 14, memory: bool = True, init: bool = True)** -> None

        Initializes the EMA indicator with the given data, period, and options.

    **init()** -> None

        Calculates the initial EMA values based on the provided data.

    **update(data: pd.Series)** -> None

        Updates the EMA based on new incoming data.

    **ema()** -> pd.Series

        Returns the stored EMA values if memory is enabled.
        Throws a MemoryError if memory=False

    **ema_latest()** -> float

        Returns the latest EMA value.
    """

    _close = np.nan

    _ema: pd.Series
    _ema_latest = np.nan

    _weight = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        weight: np.float64 = 2.0,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        """
        Initializes the EMA indicator with the given data, period, weight, and options.

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=14 | Window length for the EMA calculation.
            weight (np.float64, optional): Default=2.0 | The weight of the EMA's multiplier.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
        """
        super().__init__(data, period, memory, init)
        logger.debug(
            f"EMA init: [data_len={len(data)}, period={period}, memory={memory}]"
        )

        self._weight = weight / (period + 1)

        if init:
            self.init()

    def init(self):
        """
        Calculates the initial EMA values based on the provided data.

        This method computes the EMA using the provided data and initializes internal attributes. If memory is enabled, it also stores the computed EMA values.
        """
        close = self._data["close"]

        ema = close.ewm(
            span=self._period,
            min_periods=self._period,
            alpha=self._weight,
            adjust=False,
        ).mean()
        self._ema_latest = ema.iloc[-1]

        # Use Memory
        if self._memory:
            self._count = close.shape[0]
            self._ema = ema

        self._data = None

    def update(self, close: np.float64):
        """
        Updates the EMA based on new incoming closing price data.

        Args:
            close (np.float64): The new closing price to update the EMA with.
        """
        self._close = close
        self._ema_latest = (
            (self._close - self._ema_latest) * self._weight
        ) + self._ema_latest

        if self._memory:
            self._ema[self._count] = self._ema_latest
            self._count += 1

    def ema(self):
        """
        Returns the stored EMA values.

        Returns:
            pd.Series: A pandas Series containing the EMA values.

        Raises:
            MemoryError: If function called and memory=False
        """
        return self._ema

    def ema_latest(self):
        """
        Returns the latest EMA value.

        Returns:
            float: The most recent EMA value.
        """
        return self._ema_latest
