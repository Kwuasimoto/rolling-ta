from array import array
from typing import Dict, Literal

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _bop
from rolling_ta.indicator import Indicator


class BOP(Indicator):
    """Balance of Power"""

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int | Dict[str, int] = 14,
        memory: bool = True,
        retention: int | None = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        open = self._data["open"].to_numpy(dtype=np.float64)
        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)
        close = self._data["close"].to_numpy(dtype=np.float64)

        bop = np.zeros(close.size, dtype=np.float64)

        _bop(open, high, low, close, bop, self._period_config)

        if self._memory:
            self._bop = array("d", bop)

        self.drop_data()
        self.set_initialized()

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
