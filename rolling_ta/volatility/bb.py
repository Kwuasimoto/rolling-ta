from array import array
import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional

from rolling_ta.logging import logger

from rolling_ta.extras.numba import _bollinger_bands
from rolling_ta.trend.sma import SMA
from rolling_ta.indicator import Indicator


class BollingerBands(Indicator):
    """
    Bollinger Bands Indicator.

    Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations) which are used to
    identify volatility and potential overbought or oversold conditions in an asset.

    Material
    --------
        https://www.investopedia.com/terms/b/bollingerbands.asp
        https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands
    """

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int | Dict[str, int] = 20,
        memory: bool = True,
        retention: int | None = 20000,
        init: bool = True,
        moving_average: Optional[Indicator] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        # Use simple moving average if user does not supply a moving average.
        self._ma = (
            SMA(data, period_config, memory, retention, init)
            if moving_average is None
            else moving_average
        )

        if self._init:
            self.init()

    def init(self):
        if not self._ma._initialized:
            self._ma.init()

        close = self._data["close"].to_numpy(dtype=np.float64)
        ma = self._ma.to_numpy(dtype=np.float64)
        upper = np.zeros(ma.size, dtype=np.float64)
        lower = np.zeros(ma.size, dtype=np.float64)

        _bollinger_bands(close, ma, upper, lower, self._period_config)

        if self._memory:
            self._upper = array("d", upper)
            self._lower = array("d", lower)

        self.drop_data()
        self.set_initialized()

    def to_array(self, get: Literal["ma", "upper", "lower"] = "ma"):
        if get == "ma":
            return self._ma.to_array()
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["ma", "upper", "lower"] = "ma",
        dtype: np.dtype | None = np.float64,
    ):
        if get == "ma":
            return self._ma.to_numpy(dtype=dtype)
        return super().to_numpy(get, dtype)

    def to_series(
        self,
        get: Literal["ma", "upper", "lower"] = "ma",
        dtype: type | None = float,
        name: str | None = None,
    ):
        if get == "ma":
            return self._ma.to_series(dtype=dtype, name=name)
        return super().to_series(get, dtype, name)
