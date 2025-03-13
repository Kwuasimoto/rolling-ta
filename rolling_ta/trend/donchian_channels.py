from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _donchian_channels, _donchian_channels_update
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
        force: bool = False,
        initialization_state: bool = False,
    ) -> None:
        """The calculation of donchain is fairly opinionated, its possible to flex it but then we probably lose the simplicity of the indicator,"""
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
        if "lows" not in self._period_config:
            self._period_config.update({"lows": self._period_config["center"]})
        if "highs" not in self._period_config:
            self._period_config.update({"highs": self._period_config["center"]})
        if self._init:
            self._initialized = initialization_state
            self.calc(
                force=force,
                initialization_state=self._initialized,
            )

    def calc(self, force: bool = False, initialization_state: bool = False):
        if self._initialized and not force:
            return

        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)

        highs = np.zeros(high.size or 1, dtype=np.float64)
        lows = np.zeros(low.size or 1, dtype=np.float64)
        centers = np.zeros(high.size or 1, dtype=np.float64)

        _donchian_channels(
            high=high,
            low=low,
            highs=highs,
            lows=lows,
            centers=centers,
            period=self._period_config["center"],
        )

        self._latest_high = high[-(self._period_config["highs"] or 1) :]
        self._latest_low = low[-(self._period_config["lows"] or 1) :]

        if self._memory:
            self._highs = array("d", highs)
            self._lows = array("d", lows)
            self._center = array("d", centers)

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series) -> Indicator:
        center_latest = _donchian_channels_update(
            high=data["high"],
            low=data["low"],
            latest_high=self._latest_high,
            latest_low=self._latest_low,
            period=self._period_config["center"],
        )

        if self._memory:
            self._highs.append(self._latest_high[-1])
            self._lows.append(self._latest_low[-1])
            self._center.append(center_latest)

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
