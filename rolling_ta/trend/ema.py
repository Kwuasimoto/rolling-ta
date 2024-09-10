from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


class EMA(Indicator):

    _close = np.nan

    _ema: pd.Series
    _latest_ema = np.nan

    _weight = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        weight: np.float64 = 2.0,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        """Rolling Exponential Moving Average indicator

        https://www.investopedia.com/terms/e/ema.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=14 | Window length.
            weight (np.float64, optional): Default=2.0 | The weight of the EMA's multiplier.
            memory (bool): Default=True | Memory flag, if false removes all information not required for rsi.update().
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation
        """
        super().__init__(data, period, memory, init)
        logger.debug(
            f"EMA init: [data_len={len(data)}, period={period}, memory={memory}]"
        )

        self._weight = weight / (period + 1)

        if init:
            self.init()

    def init(self):
        close = self._data["close"]

        ema = close.ewm(
            span=self._period,
            min_periods=self._period,
            alpha=self._weight,
            adjust=False,
        ).mean()
        self._latest_ema = ema.iloc[-1]

        # Use Memory
        if self._memory:
            self._count = close.shape[0]
            self._ema = ema

        self._data = None

    def update(self, close: np.float64):
        self._close = close
        self._latest_ema = (
            (self._close - self._latest_ema) * self._weight
        ) + self._latest_ema

        if self._memory:
            self._ema[self._count] = self._latest_ema
            self._count += 1

    def ema(self):
        return self._ema

    def latest_ema(self):
        return self._latest_ema
