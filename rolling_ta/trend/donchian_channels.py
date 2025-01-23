from array import array
from typing import Dict, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _donchian_channels
from rolling_ta.indicator import Indicator


class DonchianChannels(Indicator):

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: int | Dict[str, int] = 14,
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
        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)

        highs = np.zeros(high.size, dtype=np.float64)
        lows = np.zeros(low.size, dtype=np.float64)
        centers = np.zeros(high.size, dtype=np.float64)

        _donchian_channels(high, low, highs, lows, centers, self._period_config)

        if self._memory:
            self._high = array("d", highs)
            self._low = array("d", lows)
            self._center = array("d", centers)

        if self._columns is None:
            self.set_columns()

        self.drop_data()
        self.set_initialized()

        return self

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            (
                {
                    "donchian_high": self._period_config,
                    "donchian_center": self._period_config,
                    "donchian_low": self._period_config,
                }
                if columns is None
                else columns
            ),
            name,
        )

    def to_array(
        self,
        get: Literal["high", "low", "center"] = "high",
    ):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["high", "low", "center"] = "high",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["high", "low", "center"] = "high",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
