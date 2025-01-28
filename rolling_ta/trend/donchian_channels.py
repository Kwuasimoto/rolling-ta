from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _donchian_channels
from rolling_ta.indicator import Indicator


DonchianChannelsKeys = Literal["highs", "center", "lows"]
DonchianChannelsPeriods = Literal["center"]


class DonchianChannels(Indicator):

    _keys: List[DonchianChannelsKeys] = ["highs", "center", "lows"]
    _period_config: Dict[DonchianChannelsPeriods, int] = {"center": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[DonchianChannelsKeys] = _keys,
        period_config: Dict[DonchianChannelsPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
    ) -> None:
        """The calculation of donchain is fairly opinionated, its possible to flex it but then we probably lose the simplicity of the indicator,"""
        super().__init__(
            data,
            keys=keys,
            period_config=period_config,
            memory=memory,
            retention=retention,
            init=init,
        )
        if "lows" not in self._period_config:
            self._period_config.update({"lows": self._period_config["center"]})
        if "highs" not in self._period_config:
            self._period_config.update({"highs": self._period_config["center"]})
        if self._init:
            self.calc()

    def calc(self):
        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)

        highs = np.zeros(high.size, dtype=np.float64)
        lows = np.zeros(low.size, dtype=np.float64)
        centers = np.zeros(high.size, dtype=np.float64)

        _donchian_channels(
            high=high,
            low=low,
            highs=highs,
            lows=lows,
            centers=centers,
            period=self._period_config["center"],
        )

        if self._memory:
            self._highs = array("d", highs)
            self._lows = array("d", lows)
            self._center = array("d", centers)

        self.drop_data()
        self.set_initialized()

        return self

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Dict[DonchianChannelsPeriods, int] = _period_config,
    ):
        return super().fit(data, period_config)

    def to_array(
        self,
        get: DonchianChannelsKeys = "highs",
    ):
        return super().to_array(get)

    def to_numpy(
        self,
        get: DonchianChannelsKeys = "highs",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: DonchianChannelsKeys = "highs",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
