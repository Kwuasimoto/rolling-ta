from array import array
from typing import Dict, Literal
import numpy as np
from pandas import DataFrame, Series

from rolling_ta.extras.numba import _linear_regression_r2
from rolling_ta.indicator import Indicator
from rolling_ta.trend import LinearRegression


class LinearRegressionR2(Indicator):

    _period_default = {"lr": 14, "lr2": 14}

    def __init__(
        self,
        data: DataFrame,
        period_config: int | Dict[str, int] = {"lr": 14, "lr2": 14},
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

        price = self._lr.to_numpy(get="price", dtype=np.float64)
        slopes = self._lr.to_numpy(get="slope", dtype=np.float64)
        intercepts = self._lr.to_numpy(get="intercept", dtype=np.float64)

        r2 = np.zeros(price.size, dtype=np.float64)
        _linear_regression_r2(price, slopes, intercepts, r2, self._period_config["lr2"])

        if self._memory:
            self._r2 = array("d", r2)

        self.drop_data()
        self.set_initialized()

    def update(self, data: Series):
        super().update(data, __name__)

    def to_array(self, get: Literal["r2", "slope", "intercept", "price"] = "r2"):
        if get == "slope":
            return self._lr.to_array(get)
        elif get == "intercept":
            return self._lr.to_array(get)
        elif get == "price":
            return self._lr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["r2", "slope", "intercept", "price"] = "r2",
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
        get: Literal["r2", "slope", "intercept", "price"] = "r2",
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
