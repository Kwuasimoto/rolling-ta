from array import array
from typing import Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.indicator import Indicator
from rolling_ta.extras.numba import _atr, _atr_update
from rolling_ta.volatility import TrueRange


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

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: int = 14,
        memory: bool = True,
        retention: Optional[int] = None,
        columns: Optional[list[str]] = None,
        init: bool = False,
        true_range: Optional[TrueRange] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, columns, init)
        self._tr = (
            TrueRange(data, period_config, memory, retention, columns, init)
            if true_range is None
            else true_range
        )
        self._n_1 = self._period_config - 1
        if self._init:
            self.set_columns(columns)
            self.calc()

    def calc(self):
        if not self._init:
            self._tr.calc()

        tr = self._tr.to_numpy(dtype=np.float64)
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
        self.set_initialized()

        return self

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

        return self

    def fit(self, data, period_config: Optional[int] = None):
        super().fit(data, period_config)
        self._tr.fit(data, period_config)

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            {f"atr_{self._period_config}" if columns is None else columns},
            name,
        )
        self._tr.set_columns(f"tr_{self._tr._period_config}")

    def to_array(self, get: Literal["atr", "tr"] = "atr"):
        if get == "tr":
            return self._tr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["atr", "tr"] = "atr",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        if get == "tr":
            return self._tr.to_numpy(get, dtype**kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["atr", "tr"] = "atr",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        if get == "tr":
            return self._tr.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)
