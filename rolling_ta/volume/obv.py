from array import array
from typing import Literal, Optional, Union, Dict

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _obv, _obv_update
from rolling_ta.indicator import Indicator


class OBV(Indicator):
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: Optional[int] = None,
        memory: bool = True,
        retention: Optional[int] = None,
        columns: Optional[list[str]] = None,
        init: bool = False,
    ) -> None:
        super().__init__(data, period_config, memory, retention, columns, init)
        if self._init:
            self.set_columns(columns)
            self.calc()

    def calc(self):
        close = self._data["close"].to_numpy(np.float64)
        volume = self._data["volume"].to_numpy(np.float64)
        obv = np.zeros(close.size, dtype=np.float64)

        obv, obv_latest, close_latest = _obv(close, volume, obv)

        if self._memory:
            self._obv = array("f", obv)

        self._obv_latest = obv_latest
        self._close_p = close_latest

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):
        close = data["close"]

        self._obv_latest = _obv_update(
            close, data["volume"], self._close_p, self._obv_latest
        )

        if self._memory:
            self._obv.append(self._obv_latest)

        self._close_p = close

        return self

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            f"obv_{self._period_config}" if columns is None else columns,
            name,
        )

    def to_array(self, get: Literal["obv"] = "obv"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["obv"] = "obv",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["obv"] = "obv",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
