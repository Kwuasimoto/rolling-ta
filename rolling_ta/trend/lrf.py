from array import array
from typing import Dict, Literal, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from rolling_ta.extras.numba import _linear_regression_forecast
from rolling_ta.indicator import Indicator
from rolling_ta.trend import LinearRegression


class LinearRegressionForecast(Indicator):

    _period_default = {"lr": 14, "lrf": 14}

    def __init__(
        self,
        data: DataFrame,
        period_config: int | Dict[str, int] = {"lr": 14, "lrf": 14},
        memory: bool = True,
        retention: int | None = 20000,
        init: bool = True,
        lr: LinearRegression | None = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        self._lr = (
            LinearRegression(data, period_config["lr"], memory, retention, init)
            if lr is None
            else lr
        )

        if self._init:
            self.init()

    def init(self):
        if not self._lr._initialized:
            self._lr.init()

        slopes = self._lr.to_numpy(get="slope", dtype=np.float64)
        intercepts = self._lr.to_numpy(get="intercept", dtype=np.float64)

        forecast = np.zeros(slopes.size + self._period_config["lrf"], dtype=np.float32)
        _linear_regression_forecast(
            slopes, intercepts, forecast, self._period_config["lrf"]
        )

        if self._memory:
            self._forecast = array("d", forecast)

        self.drop_data()
        self.set_initialized()

    def update(self, data: Series):
        super().update(data, __name__)

    def to_array(
        self, get: Literal["forecast", "slope", "intercept", "price"] = "forecast"
    ):
        if get == "slope":
            return self._lr.to_array(get)
        elif get == "intercept":
            return self._lr.to_array(get)
        elif get == "price":
            return self._lr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["forecast", "slope", "intercept", "price"] = "forecast",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        if get == "slope":
            return self._lr.to_numpy(get, dtype, **kwargs)
        elif get == "intercept":
            return self._lr.to_numpy(get, dtype, **kwargs)
        elif get == "price":
            return self._lr.to_numpy(get, dtype, **kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["forecast", "slope", "intercept", "price"] = "forecast",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        if get == "slope":
            return self._lr.to_series(get, dtype, name, **kwargs)
        elif get == "intercept":
            return self._lr.to_series(get, dtype, name, **kwargs)
        elif get == "price":
            return self._lr.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)
