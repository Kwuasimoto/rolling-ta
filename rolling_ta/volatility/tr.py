from array import array
from typing import Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _tr, _tr_update
from rolling_ta.indicator import Indicator

from rolling_ta.logging import log


class TrueRange(Indicator):

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: int = 14,
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
        if self._columns is None:
            self.set_columns()

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
            self._tr = array("d", tr)

        self._tr_latest = tr_latest
        self._close_p = close_p

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series) -> np.float64:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        self._tr_latest = _tr_update(high, low, self._close_p)

        if self._memory:
            self._tr.append(self._tr_latest)

        self._close_p = close

        return self._tr_latest

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            f"tr_{self._period_config}" if columns is None else columns,
            name,
        )

    def to_array(self, get: Literal["tr"] = "tr"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["tr"] = "tr",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["tr"] = "tr",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)

    def tr_latest(self):
        return self._tr_latest
