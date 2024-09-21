from array import array
from rolling_ta.extras.numba import _obv, _obv_update
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
from typing import Union, Dict

import pandas as pd
import numpy as np


class OBV(Indicator):
    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Dict[str, int] = {"ema": 20},
        memory: bool = True,
        retention: Union[int, None] = None,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        if self._init:
            self.init()

    def init(self):
        close = self._data["close"].to_numpy(np.float64)
        volume = self._data["volume"].to_numpy(np.float64)
        obv = np.zeros(close.size, dtype=np.float64)

        obv, obv_latest, close_latest = _obv(close, volume, obv)

        if self._memory:
            self._obv = array("f", obv)

        self._obv_latest = obv_latest
        self._close_p = close_latest

        self.drop_data()

    def update(self, data: pd.Series):
        close = data["close"]

        self._obv_latest = _obv_update(
            close, data["volume"], self._close_p, self._obv_latest
        )

        if self._memory:
            self._obv.append(self._obv_latest)

        self._close_p = close

    def obv(self):
        return pd.Series(self._obv)
