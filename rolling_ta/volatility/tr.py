from array import array
from typing import Union
from rolling_ta.extras.numba import _tr, _tr_update
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
import pandas as pd
import numpy as np


class NumbaTrueRange(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: Union[int, None] = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        if self._init:
            self.init()

    def init(self):
        high = self._data["high"].to_numpy(np.float64)
        low = self._data["low"].to_numpy(np.float64)
        close = self._data["close"].to_numpy(np.float64)

        close_p = np.zeros(close.size, dtype=np.float64)
        tr = np.zeros(close.size, dtype=np.float64)

        tr, tr_latest, close_p = _tr(high, low, close, close_p, tr)

        # Save numpy copy for indicators that depend on tr
        self._tr = tr

        # If memory set, convert to array
        if self._memory:
            self._tr = array("f", tr)

        self._tr_latest = tr_latest
        self._close_p = close_p

        self.drop_data()

    def update(self, data: pd.Series) -> np.float64:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        self._tr_latest = _tr_update(high, low, self._close_p)

        if self._memory:
            self._tr.append(self._tr_latest)

        self._close_p = close
        return self._tr_latest

    def tr(self):
        return pd.Series(self._tr)

    def tr_latest(self):
        return self._tr_latest


class TrueRange(Indicator):

    _tr: pd.Series
    _tr_latest = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

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

        if self._memory:
            self._tr = tr
            self._count = close.shape[0]

        self._close_p = close.iloc[-1]

        self.drop_data()

    def update(self, data: pd.Series):
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr = np.max(
            [high - low, np.abs(high - self._close_p), np.abs(low - self._close_p)]
        )

        if self._memory:
            self._tr[self._count] = tr
            self._count += 1

        self._tr_latest = tr
        self._close_p = close

    def tr(self):
        if not self._memory:
            raise MemoryError("TrueRange._memory = False")
        return self._tr

    def tr_latest(self):
        return self._tr_latest
