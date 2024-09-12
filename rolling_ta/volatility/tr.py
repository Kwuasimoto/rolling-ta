from rolling_ta.indicator import Indicator

import pandas as pd
import numpy as np


class TrueRange(Indicator):

    _tr: pd.Series
    _tr_latest = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period, memory, retention, init)

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
