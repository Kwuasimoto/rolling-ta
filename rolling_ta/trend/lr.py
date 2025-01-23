from array import array
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _typical_price, _linear_regression
from rolling_ta.indicator import Indicator


class LinearRegression(Indicator):

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
        close = self._data["close"].to_numpy(dtype=np.float64)

        price = np.empty(close.size, dtype=np.float64)
        _typical_price(high, low, close, price)

        slope = np.zeros(price.size, dtype=np.float32)
        intercept = np.zeros(price.size, dtype=np.float32)
        _linear_regression(price, slope, intercept, self._period_config)

        if self._memory:
            self._price = array("d", price)
            self._slope = array("d", slope)
            self._intercept = array("d", intercept)

        if self._columns is None:
            self.set_columns()

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):
        super().update(data, __name__)

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            (
                {"lr": self._period_config, "lrs": self._period_config}
                if columns is None
                else columns
            ),
            name,
        )

    def to_array(self, get: Literal["slope", "intercept", "price"] = "slope"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["slope", "intercept", "price"] = "slope",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["slope", "intercept", "price"] = "slope",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
