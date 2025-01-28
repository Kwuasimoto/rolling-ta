from array import array
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _linear_regression_r2
from rolling_ta.indicator import Indicator

from .lr import LinearRegression, LinearRegressionKeys, LinearRegressionPeriods


LinearRegressionR2Keys = Union[Literal["lr2"], LinearRegressionKeys]
LinearRegressionR2Periods = Union[Literal["lr2"], LinearRegressionPeriods]


class LinearRegressionR2(Indicator):

    _keys: List[LinearRegressionR2Keys] = ["lr2", "price", "slope", "intercept"]
    _period_default: Dict[LinearRegressionR2Periods, int] = {
        "price": 14,
        "lr2": 14,
    }

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[LinearRegressionR2Keys] = _keys,
        period_config: Dict[LinearRegressionR2Periods, int] = _period_default,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        lr: Optional[LinearRegression] = None,
    ) -> None:
        super().__init__(data, keys, period_config, memory, retention, init)
        if "price" not in self._period_config:
            self._period_config["price"] = self._period_config["lr2"]
        if "intercept" not in self._period_config:
            self._period_config["intercept"] = self._period_config["price"]
        if "slope" not in self._period_config:
            self._period_config["slope"] = self._period_config["price"]
        self._lr = (
            LinearRegression(
                data,
                keys=["intercept", "price", "slope"],
                period_config={"price": self._period_config["price"]},
                memory=memory,
                retention=retention,
                init=init,
            )
            if lr is None
            else lr
        )
        if self._init:
            self.calc()

    def calc(self):
        if not self._lr._initialized:
            self._lr.calc()

        price = self._lr.to_numpy(get="price", dtype=np.float64)
        slopes = self._lr.to_numpy(get="slope", dtype=np.float64)
        intercepts = self._lr.to_numpy(get="intercept", dtype=np.float64)

        lr2 = np.zeros(price.size, dtype=np.float64)
        _linear_regression_r2(
            ys=price,
            slopes=slopes,
            intercepts=intercepts,
            r2_container=lr2,
            period=self._period_config["lr2"],
        )

        if self._memory:
            self._lr2 = array("d", lr2)

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):
        super().update(data, __name__)

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Dict[LinearRegressionR2Periods, int] = _period_default,
    ):
        super().fit(data, period_config)
        if set(period_config.keys()) & (self._lr._period_config.keys()):
            self._lr.fit(data, period_config={"price": period_config["price"]})

    def to_array(self, get: LinearRegressionR2Keys = "lr2"):
        if get == "slope":
            return self._lr.to_array(get)
        elif get == "intercept":
            return self._lr.to_array(get)
        elif get == "price":
            return self._lr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: LinearRegressionR2Keys = "lr2",
        dtype: Optional[np.dtype] = np.float64,
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
        get: LinearRegressionR2Keys = "lr2",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        if get == "slope":
            return self._lr.to_series(get, dtype, name, **kwargs)
        elif get == "intercept":
            return self._lr.to_series(get, dtype, name, **kwargs)
        elif get == "price":
            return self._lr.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)
