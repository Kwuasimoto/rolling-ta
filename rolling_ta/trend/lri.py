from typing import Dict
import numpy as np
from pandas import DataFrame, Series
from rolling_ta.extras.numba import _typical_price
from rolling_ta.indicator import Indicator


class LinearRegressionIntercept(Indicator):

    def __init__(
        self,
        data: DataFrame,
        period_config: int | Dict[str, int],
        memory: bool,
        retention: int | None,
        init: bool,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

    def init(self):
        typical_price = np.empty(self._data["close"].size, dtype=np.float64)
        _typical_price(self._data["high"], self._data["low"], self._data["low"])

        self.drop_data()

    def update(self, data: Series):
        return super().update(data)
