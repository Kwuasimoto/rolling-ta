from array import array
from pandas import DataFrame
from rolling_ta.extras.numba import (
    _mf_pos_neg,
    _mf_pos_neg_sum,
    _mfi,
    _mfi_update,
    _mf_update,
    _rmf,
    _typical_price,
    _typical_price_single,
)

from rolling_ta.indicator import Indicator
import pandas as pd
import numpy as np

from collections import deque


class MFI(Indicator):
    """
    Money Flow Index (MFI) indicator.

    The MFI is a momentum indicator that uses both price and volume data to
    identify overbought or oversold conditions in an asset. This class calculates
    the MFI using historical price and volume data over a specified period.

    Material
    --------
     - https://www.investopedia.com/terms/m/mfi.asp
     - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/money-flow-index-mfi
     - https://pypi.org/project/ta/
    """

    def __init__(
        self,
        data: DataFrame,
        period: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        high = self._data["high"].to_numpy(np.float64)
        low = self._data["low"].to_numpy(np.float64)
        close = self._data["close"].to_numpy(np.float64)
        volume = self._data["volume"].to_numpy(np.float64)

        typical_price = np.zeros(volume.size, dtype=np.float64)
        _typical_price(high, low, close, typical_price)

        rmf = np.zeros(volume.size, dtype=np.float64)
        _rmf(typical_price, volume, rmf)

        pmf = np.zeros(volume.size, dtype=np.float64)
        nmf = np.zeros(volume.size, dtype=np.float64)
        _mf_pos_neg(typical_price, rmf, pmf, nmf)

        pmf_sums = np.zeros(volume.size, dtype=np.float64)
        nmf_sums = np.zeros(volume.size, dtype=np.float64)
        _mf_pos_neg_sum(pmf, nmf, pmf_sums, nmf_sums, self._period_config)

        mfi = np.zeros(volume.size, dtype=np.float64)
        _mfi(pmf_sums, nmf_sums, mfi, self._period_config)

        if self._memory:
            self._mfi = array("f", mfi)

        self._typical_price_prev = typical_price[-1]

        self._pmf_sum = pmf_sums[-1]
        self._nmf_sum = nmf_sums[-1]
        self._pmf_window = pmf[-self._period_config :]
        self._nmf_window = nmf[-self._period_config :]

        self.drop_data()

    def update(self, data: pd.Series):
        volume = data["volume"]
        typical_price = _typical_price_single(data["high"], data["low"], data["close"])

        self._pmf_sum, self._nmf_sum = _mf_update(
            volume,
            typical_price,
            self._typical_price_prev,
            self._pmf_window,
            self._nmf_window,
            self._pmf_sum,
            self._nmf_sum,
        )

        mfi = _mfi_update(self._pmf_sum, self._nmf_sum)

        self._typical_price_prev = typical_price

        if self._memory:
            self._mfi.append(mfi)

    def mfi(self):
        return pd.Series(self._mfi)
