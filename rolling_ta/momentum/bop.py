from array import array
from typing import Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _bop
from rolling_ta.indicator import Indicator

from rolling_ta.logging import log


class BOP(Indicator):
    """Balance of Power"""

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
        open = self._data["open"].to_numpy(dtype=np.float64)
        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)
        close = self._data["close"].to_numpy(dtype=np.float64)

        bop = np.zeros(close.size, dtype=np.float64)

        _bop(open, high, low, close, bop, self._period_config)

        if self._memory:
            self._bop = array("d", bop)

        if self._columns is None:
            self.set_columns()

        self.drop_data()
        self.set_initialized()

        return self

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            f"bop_{self._period_config}" if columns is None else columns,
            name,
        )

    def to_array(self, get: Literal["bop"] = "bop"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["bop"] = "bop",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["bop"] = "bop",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
