from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _tr, _tr_update
from rolling_ta.indicator import Indicator


TrueRangeKeys = Literal["tr"]
TrueRangePeriods = Literal["tr"]


class TrueRange(Indicator):

    _keys: List[TrueRangeKeys] = ["tr"]
    _period_config: Dict[TrueRangePeriods, int] = {"tr": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[TrueRangeKeys] = _keys,
        period_config: Dict[TrueRangePeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        force: bool = False,
        initialization_state: bool = False,
    ) -> None:
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
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: Optional[bool] = True):
        if self._initialized and not force:
            return

        high = self._data["high"].to_numpy(np.float64)
        low = self._data["low"].to_numpy(np.float64)
        close = self._data["close"].to_numpy(np.float64)

        close_p = np.zeros(close.size, dtype=np.float64)
        tr = np.zeros(close.size, dtype=np.float64)

        tr, tr_latest, close_p = _tr(
            high=high,
            low=low,
            close=close,
            close_p_container=close_p,
            tr_container=tr,
        )

        # Save numpy copy for indicators that depend on tr
        self._tr = tr

        # If memory set, convert to array
        if self._memory:
            self._tr = array("d", tr)

        self._tr_latest = tr_latest
        self._close_p = close_p

        self.drop_data()
        self.set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series) -> np.float64:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        self._tr_latest = _tr_update(high, low, self._close_p)

        if self._memory:
            self._tr.append(self._tr_latest)

        self._close_p = close
        return self._tr_latest

    def fit(self, data, period_config: Dict[TrueRangePeriods, int] = _period_config):
        return super().fit(data, period_config)

    def to_array(self, get: TrueRangeKeys = "tr"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: TrueRangeKeys = "tr",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: TrueRangeKeys = "tr",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)

    def tr_latest(self):
        return self._tr_latest
