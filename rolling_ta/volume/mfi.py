from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

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

MFIKeys = Literal["mfi"]
MFIPeriods = Literal["mfi"]


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

    _keys: List[MFIKeys] = ["mfi"]
    _period_config: Dict[MFIPeriods, int] = {"mfi": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[MFIKeys] = _period_config,
        period_config: Dict[MFIPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
    ) -> None:
        super().__init__(
            data,
            keys=keys,
            period_config=period_config,
            memory=memory,
            retention=retention,
            init=init,
        )
        if self._init:
            self.calc()

    def calc(self):
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
        _mf_pos_neg_sum(pmf, nmf, pmf_sums, nmf_sums, self._period_config["mfi"])

        mfi = np.zeros(volume.size, dtype=np.float64)
        _mfi(pmf_sums, nmf_sums, mfi, self._period_config["mfi"])

        if self._memory:
            self._mfi = array("f", mfi)

        self._typical_price_prev = typical_price[-1]

        self._pmf_sum = pmf_sums[-1]
        self._nmf_sum = nmf_sums[-1]
        self._pmf_window = pmf[-self._period_config["mfi"] :]
        self._nmf_window = nmf[-self._period_config["mfi"] :]

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):
        volume = data["volume"]
        typical_price = _typical_price_single(data["high"], data["low"], data["close"])

        self._pmf_sum, self._nmf_sum = _mf_update(
            volume=volume,
            price_curr=typical_price,
            price_prev=self._typical_price_prev,
            pmf_window=self._pmf_window,
            nmf_window=self._nmf_window,
            pmf_sum=self._pmf_sum,
            nmf_sum=self._nmf_sum,
        )

        mfi = _mfi_update(pmf_sum=self._pmf_sum, nmf_sum=self._nmf_sum)

        self._typical_price_prev = typical_price

        if self._memory:
            self._mfi.append(mfi)

        return self

    def fit(self, data, period_config: MFIPeriods = _period_config):
        return super().fit(data, period_config)

    def to_array(self, get: MFIKeys = "mfi"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: MFIKeys = "mfi",
        dtype: np.dtype = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: MFIKeys = "mfi",
        dtype: type = float,
        name: str = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
