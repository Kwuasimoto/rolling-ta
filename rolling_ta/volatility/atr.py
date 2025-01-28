from array import array
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
import numpy as np

from rolling_ta.indicator import Indicator
from rolling_ta.extras.numba import _atr, _atr_update

from .tr import TrueRange, TrueRangeKeys, TrueRangePeriods


AverageTrueRangeKeys = Union[Literal["atr"], TrueRangeKeys]
AverageTrueRangePeriods = Union[Literal["atr"], TrueRangePeriods]


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
    - https://www.investopedia.com/terms/a/atr.asp
    - https://pypi.org/project/ta/
    """

    _keys: List[AverageTrueRangeKeys] = ["atr", "tr"]
    _period_config: Dict[AverageTrueRangePeriods, int] = {"atr": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[AverageTrueRangeKeys] = _keys,
        period_config: Dict[AverageTrueRangePeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        true_range: Optional[TrueRange] = None,
    ) -> None:
        super().__init__(data, keys, period_config, memory, retention, init)
        if "tr" not in self._period_config:
            self._period_config.update({"tr": self._period_config["atr"]})
        self._tr = (
            TrueRange(
                data,
                keys=["tr"],
                period_config={"tr": self._period_config["tr"]},
                memory=memory,
                retention=retention,
                init=init,
            )
            if true_range is None
            else true_range
        )
        self._p_1 = (
            self._period_config["atr"] - 1
            if "p_1" not in period_config
            else period_config["p_1"]
        )
        if self._init:
            self.calc()

    def calc(self):
        if not self._init:
            self._tr.calc()

        tr = self._tr.to_numpy(dtype=np.float64)
        atr = np.zeros(tr.size, dtype=np.float64)

        self._atr, latest = _atr(
            tr=tr,
            atr_container=atr,
            period=self._period_config["atr"],
            p_1=self._p_1,
        )

        self._atr_latest = latest

        if self._memory:
            self._atr = array("d", self._atr)

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):

        self._tr.update(data)

        self._atr_latest = _atr_update(
            atr_latest=self._atr_latest,
            tr_current=self._tr._tr_latest,
            period=self._period_config["atr"],
            p_1=self._p_1,
        )

        if self._memory:
            self._atr.append(self._atr_latest)
        return self

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Dict[AverageTrueRangePeriods, int] = _period_config,
    ):
        super().fit(data, period_config)
        if set(period_config.keys()) & set(self._tr._period_config.keys()):
            self._tr.fit(data, period_config)

    def to_array(self, get: AverageTrueRangeKeys = "atr"):
        if get == "tr":
            return self._tr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: AverageTrueRangeKeys = "atr",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        if get == "tr":
            return self._tr.to_numpy(get, dtype, **kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: AverageTrueRangeKeys = "atr",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        if get == "tr":
            return self._tr.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)
