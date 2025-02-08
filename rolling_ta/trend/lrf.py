from array import array
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _linear_regression_forecast
from rolling_ta.indicator import Indicator

from .lr import LinearRegression, LinearRegressionKeys, LinearRegressionPeriods


LinearRegressionForecastKeys = Union[Literal["lrf"], LinearRegressionKeys]
LinearRegressionForecastPeriods = Union[Literal["lrf"], LinearRegressionPeriods]


class LinearRegressionForecast(Indicator):

    _keys: List[LinearRegressionForecastKeys] = [
        "lr",
        "lrf",
        "price",
        "slope",
        "intercept",
    ]
    _period_default: Dict[LinearRegressionForecastPeriods, int] = {
        "price": 14,
        "lrf": 14,
    }

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[LinearRegressionForecastKeys] = _keys,
        period_config: Dict[LinearRegressionForecastPeriods, int] = _period_default,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        lr: Optional[LinearRegression] = None,
        force: bool = False,
        initialization_state: bool = False,
    ) -> None:
        super().__init__(
            data=data,
            keys=keys,
            period_config=period_config,
            memory=memory,
            retention=retention,
            init=init,
            force=force,
            initialization_state=initialization_state,
        )
        if "price" not in self._period_config:
            self._period_config["price"] = self._period_config["lrf"]
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
                force=force,
                initialization_state=initialization_state,
            )
            if lr is None
            else lr
        )
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: Optional[bool] = True):
        if self._initialized and not force:
            return

        if not self._lr._initialized or force:
            self._lr.calc(
                force=force,
                initialization_state=initialization_state,
            )

        slopes = self._lr.to_numpy(get="slope", dtype=np.float64)
        intercepts = self._lr.to_numpy(get="intercept", dtype=np.float64)

        forecast = np.zeros(slopes.size + self._period_config["lrf"], dtype=np.float32)
        _linear_regression_forecast(
            slopes, intercepts, forecast, self._period_config["lrf"]
        )

        if self._memory:
            self._lrf = array("d", forecast)

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series):
        super().update(data, __name__)

    def fit(self, data, period_config: Optional[Dict[str, int]] = _period_default):
        super().fit(data, period_config)
        if set(period_config.keys()) & (self._lr._period_config.keys()):
            self._lr.fit(data, period_config={"price": period_config["price"]})

    def to_array(self, get: LinearRegressionForecastKeys = "lrf"):
        if get == "slope":
            return self._lr.to_array(get)
        elif get == "intercept":
            return self._lr.to_array(get)
        elif get == "price":
            return self._lr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: LinearRegressionForecastKeys = "lrf",
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
        get: LinearRegressionForecastKeys = "lrf",
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
