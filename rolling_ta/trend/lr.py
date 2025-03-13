from array import array
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import (
    _typical_price,
    _linear_regression,
    _linear_regression_update,
)
from rolling_ta.indicator import Indicator


LinearRegressionKeys = Literal["lr", "price", "slope", "intercept"]
LinearRegressionPeriods = Literal["price"]


class LinearRegression(Indicator):

    _keys: List[LinearRegressionKeys] = ["lr", "price", "slope", "intercept"]
    _period_config: Dict[LinearRegressionPeriods, int] = {"price": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[LinearRegressionKeys] = _keys,
        period_config: Dict[LinearRegressionPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        force: bool = False,
        initialization_state: bool = False,
    ) -> None:
        """period_config is rather opinionated, might try to make it more flexible in the future.

        Use price key for period config.
        """
        super().__init__(
            data,
            keys=keys,
            period_config=period_config,
            memory=memory,
            retention=retention,
            init=init,
            force=force,
            initialization_state=initialization_state,
        )
        if "slope" not in self._period_config:
            self._period_config.update({"slope": self._period_config["price"]})
        if "intercept" not in self._period_config:
            self._period_config.update({"intercept": self._period_config["price"]})
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: Optional[bool] = True):
        if self._initialized and not force:
            return

        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)
        close = self._data["close"].to_numpy(dtype=np.float64)

        price = np.empty(close.size, dtype=np.float64)

        self.x = 0.0
        self.xx = 0.0

        for i in range(self._period_config["price"]):
            self.x += i
            self.xx += i * i

        _typical_price(
            high=high,
            low=low,
            close=close,
            price_container=price,
        )

        slope = np.zeros(price.size, dtype=np.float32)
        intercept = np.zeros(price.size, dtype=np.float32)
        _linear_regression(
            ys=price,
            slope_container=slope,
            intercept_container=intercept,
            period=self._period_config["price"],
            x=self.x,
            xx=self.xx,
        )

        self._y_latest = price[-self._period_config["price"] :]

        if self._memory:
            self._price = array("d", price)
            self._slope = array("d", slope)
            self._intercept = array("d", intercept)

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series):
        high = data["high"]
        low = data["low"]
        close = data["close"]
        price = (high + low + close) / 3

        slope, intercept = _linear_regression_update(
            price=price,
            y_latest=self._y_latest,
            period=self._period_config["price"],
            x=self.x,
            xx=self.xx,
        )

        if self._memory:
            self._price = price
            self._slope.append(slope)
            self._intercept.append(intercept)

        return self

    def get(self, index: int, key: LinearRegressionKeys = "price"):
        return super().get(index, key)

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Dict[LinearRegressionPeriods, int] = _period_config,
    ):
        return super().fit(data, period_config)

    def to_array(self, get: LinearRegressionKeys = "slope"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: LinearRegressionKeys = "slope",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: LinearRegressionKeys = "slope",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
