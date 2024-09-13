from rolling_ta.indicator import Indicator
from typing import Union, Dict

import pandas as pd
import numpy as np


class OBV(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Dict[str, int] = {"ema": 20},
        memory: bool = True,
        retention: Union[int, None] = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        close = self._data["close"]
        volume = self._data["volume"]

        close_p = close.shift(1)
        obv = np.zeros_like(volume)

        obv_p_mask = close > close_p
        obv_n_mask = close < close_p

        obv[obv_p_mask] = volume[obv_p_mask]
        obv[obv_n_mask] = -volume[obv_n_mask]

        obv = pd.Series(obv, index=volume.index).cumsum()

        if self._memory:
            self._obv = obv
            self._count = obv.shape[0]

        self._obv_latest = obv.iat[-1]
        self._close_p = close.iat[-1]

        self.drop_data()

    def update(self, data: pd.Series):

        close = data["close"]
        volume = data["volume"]

        if close > self._close_p:
            self._obv_latest += volume
        elif close < self._close_p:
            self._obv_latest -= volume

        if self._memory:
            self._obv[self._count] = self._obv_latest
            self._count += 1

        self._close_p = close

    def obv(self):
        return self._obv

    def obv_latest(self):
        return self._obv_latest
