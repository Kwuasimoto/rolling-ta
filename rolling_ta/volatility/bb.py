from array import array
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _bollinger_bands, _bollinger_bands_update
from rolling_ta.trend.sma import SMA
from rolling_ta.indicator import Indicator
from rolling_ta.logging import log


BollingerBandsKeys = Literal["bb", "ma"]
BollingerBandsPeriods = Union[Literal["weight"], BollingerBandsKeys]


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

    _keys: List[BollingerBandsKeys] = ["upper", "lower", "ma"]
    _period_config: Dict[BollingerBandsPeriods, int] = {"bb": 20, "ma": 20, "weight": 2}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[BollingerBandsKeys] = _keys,
        period_config: Dict[BollingerBandsPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        moving_average: Optional[Indicator] = None,
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

        # Use simple moving average if user does not supply a moving average.
        if "ma" not in self._period_config:
            self._period_config["ma"] = self._period_config["bb"]

        # TODO
        # This is a problem in multiple indicators, will need to be refactored eventually:
        # The problem is coupling the period config to a indicator object in memory
        # This needs to be dealt with sooner than later.
        if "upper" not in self._period_config:
            self._period_config["upper"] = self._period_config["ma"]
        if "lower" not in self._period_config:
            self._period_config["lower"] = self._period_config["ma"]

        self._ma = (
            SMA(
                data,
                keys=["sma"],
                period_config={"sma": self._period_config["ma"]},
                memory=memory,
                retention=retention,
                init=init,
                force=force,
                initialization_state=initialization_state,
            )
            if moving_average is None
            else moving_average
        )

        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: Optional[bool] = True):
        """Performs early return if _initialized is True"""
        if self._initialized and not force:
            return

        if not self._ma._initialized or force:
            self._ma.calc(force=force, initialization_state=initialization_state)

        close = self._data["close"].to_numpy(dtype=np.float64)
        ma = self._ma.to_numpy(dtype=np.float64)
        upper = np.zeros(ma.size, dtype=np.float64)
        lower = np.zeros(ma.size, dtype=np.float64)

        _bollinger_bands(
            price=close,
            ma=ma,
            upper_container=upper,
            lower_container=lower,
            period=self._period_config["bb"],
            weight=self._period_config["weight"],
        )

        self._price_latest = close[-self._period_config["bb"] :]

        if self._memory:
            self._upper = array("d", upper)
            self._lower = array("d", lower)

        self.drop_data()
        self.set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series) -> Indicator:
        self._ma.update(data)

        upper, lower = _bollinger_bands_update(
            price=data["close"],
            price_latest=self._price_latest,
            ma=self._ma.get(-1),
            period=self._period_config["bb"],
            weight=self._period_config["weight"],
        )

        if self._memory:
            self._upper.append(upper)
            self._lower.append(lower)

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Dict[BollingerBandsPeriods, int] = _period_config,
    ):
        super().fit(data, period_config)
        self._ma.fit(data, period_config)

    def to_array(self, get: BollingerBandsKeys = "ma"):
        if get == "ma":
            return self._ma.to_array()
        return super().to_array(get)

    def to_numpy(
        self,
        get: BollingerBandsKeys = "ma",
        dtype: Optional[np.dtype] = np.float64,
    ):
        if get == "ma":
            return self._ma.to_numpy(dtype=dtype)
        return super().to_numpy(get, dtype)

    def to_series(
        self,
        get: BollingerBandsKeys = "ma",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
    ):
        if get == "ma":
            return self._ma.to_series(dtype=dtype, name=name)
        return super().to_series(get, dtype, name)
