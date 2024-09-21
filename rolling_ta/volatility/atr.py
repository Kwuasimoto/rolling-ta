from array import array
from pandas import DataFrame
from rolling_ta.indicator import Indicator
from rolling_ta.extras.numba import _atr, _atr_update
from rolling_ta.volatility import TR


import pandas as pd
import numpy as np

from typing import Optional


class ATR(Indicator):
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
    """

    def __init__(
        self,
        data: DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
        true_range: Optional[TR] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        self._tr = TR(data, period_config) if true_range is None else true_range
        self._n_1 = self._period_config - 1
        if self._init:
            self.init()

    def init(self):
        if not self._init:
            self._tr.init()

        tr = self._tr.tr().to_numpy(dtype=np.float64)
        atr = np.zeros(tr.size, dtype=np.float64)

        self._atr, latest = _atr(
            tr,
            atr,
            self._period_config,
            self._n_1,
        )

        self._atr_latest = latest

        if self._memory:
            self._atr = array("d", self._atr)

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
            self._atr.append(self._atr_latest)

    def atr(self):
        return pd.Series(self._atr)
