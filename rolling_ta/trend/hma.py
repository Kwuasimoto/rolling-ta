from array import array
from typing import Dict, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _hma
from rolling_ta.trend.wma import WMA
from rolling_ta.indicator import Indicator


class HMA(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int | Dict[str, int] = 14,
        memory: bool = True,
        retention: int | None = 20000,
        init: bool = True,
        wma_full: Optional[WMA] = None,
        wma_half: Optional[WMA] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        self._wma_full = (
            WMA(data, period_config, memory, retention, init)
            if wma_full is None
            else wma_full
        )
        self._wma_half = (
            WMA(data, period_config // 2, memory, retention, init)
            if wma_half is None
            else wma_half
        )

        if self._init:
            self.init()

    def init(self):
        if not self._wma_full._initialized:
            self._wma_full.init()
        if not self._wma_half._initialized:
            self._wma_half.init()

        close = self._data["close"].to_numpy(dtype=np.float64)

        wma_full = self._wma_full.to_numpy()
        wma_half = self._wma_half.to_numpy()

        hma_internim = np.zeros(close.size, dtype=np.float64)
        hma = np.zeros(close.size, dtype=np.float64)

        _hma(wma_full, wma_half, hma_internim, hma, self._period_config)

        if self._memory:
            self._hma = array("d", hma)

        self.drop_data()
        self.set_initialized()

    def to_array(self, get: Literal["hma"] = "hma"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["hma"] = "hma",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["hma"] = "hma",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
