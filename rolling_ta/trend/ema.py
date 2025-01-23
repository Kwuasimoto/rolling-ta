from array import array
from typing import Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _ema, _ema_update
from rolling_ta.indicator import Indicator


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) Indicator.

    The EMA gives more weight to recent prices, making it more responsive to new information compared to the Simple Moving Average (SMA).
    This indicator is commonly used to identify trends and smooth out price data.

    Material
    --------
        https://www.investopedia.com/terms/e/ema.asp
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: int = 14,
        memory: bool = True,
        retention: Optional[int] = None,
        columns: Optional[list[str]] = None,
        init: bool = False,
        weight: np.float64 = 2.0,
    ) -> None:
        super().__init__(data, period_config, memory, retention, columns, init)
        self._weight = weight / (period_config + 1)
        if self._init:
            self.set_columns(columns)
            self.calc()

    def calc(self):
        close = self._data["close"].to_numpy(dtype=np.float64)

        ema = np.zeros(close.size)
        ema, ema_latest = _ema(
            close,
            ema,
            self._weight,
            self._period_config,
        )

        self._ema_latest = ema_latest

        if self._memory:
            self._ema = array("d", ema)

        if self._columns is None:
            self.set_columns()

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):
        self._ema_latest = _ema_update(data["close"], self._weight, self._ema_latest)

        if self._memory:
            self._ema.append(self._ema_latest)

        return self

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            f"ema_{self._period_config}" if columns is None else columns,
            name,
        )

    def to_array(self, get: Literal["ema"] = "ema"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["ema"] = "ema",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["ema"] = "ema",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
