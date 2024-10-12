from array import array
from typing import Dict, Literal, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from rolling_ta.extras.numba import _typical_price, _linear_regression
from rolling_ta.indicator import Indicator


class LinearRegression(Indicator):

    def __init__(
        self,
        data: DataFrame,
        period_config: int | Dict[str, int] = 14,
        memory: bool = True,
        retention: int | None = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)
        close = self._data["close"].to_numpy(dtype=np.float64)

        price = np.empty(close.size, dtype=np.float64)
        _typical_price(high, low, close, price)

        slope = np.zeros(price.size, dtype=np.float32)
        intercept = np.zeros(price.size, dtype=np.float32)
        _linear_regression(price, slope, intercept, self._period_config)

        if self._memory:
            self._slope = array("d", slope)
            self._intercept = array("d", intercept)
            self._price = array("d", price)

        self.drop_data()
        self.set_initialized()

    def update(self, data: Series):
        super().update(data, __name__)

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
