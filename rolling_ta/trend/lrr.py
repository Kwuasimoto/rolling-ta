from array import array
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _linear_regression_r2
from rolling_ta.indicator import Indicator
from rolling_ta.trend import LinearRegression


class LinearRegressionR2(Indicator):

    _period_default = {"lr": 14, "lr2": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: int | Dict[str, int] = {"lr": 14, "lr2": 14},
        memory: bool = True,
        retention: Optional[int] = None,
        columns: Optional[list[str]] = None,
        init: bool = False,
        lr: Optional[LinearRegression] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, columns, init)

        self._lr = (
            LinearRegression(
                data, period_config["lr"], memory, retention, columns, init
            )
            if lr is None
            else lr
        )

        if self._init:
            self.set_columns(columns)
            self.calc()

    def calc(self):
        if not self._lr._initialized:
            self._lr.calc()

        price = self._lr.to_numpy(get="price", dtype=np.float64)
        slopes = self._lr.to_numpy(get="slope", dtype=np.float64)
        intercepts = self._lr.to_numpy(get="intercept", dtype=np.float64)

        r2 = np.zeros(price.size, dtype=np.float64)
        _linear_regression_r2(price, slopes, intercepts, r2, self._period_config["lr2"])

        if self._memory:
            self._r2 = array("d", r2)

        if self._columns is None:
            self.set_columns()

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):
        super().update(data, __name__)

    def fit(self, data):
        super().fit(data)
        self._lr.fit(data)

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            f"lr2_{self._period_config['lr2']}" if columns is None else columns, name
        )
        self._lr.set_columns(f"lr_{self._lr._period_config}")

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
