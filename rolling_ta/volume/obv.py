from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _obv, _obv_update
from rolling_ta.indicator import Indicator

OBVKeys = Literal["obv"]
OBVPeriods = Literal["obv"]


class OBV(Indicator):

    _keys: List[OBVKeys] = ["obv"]
    _period_config: Dict[OBVPeriods, int] = {"obv": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[OBVKeys] = _keys,
        period_config: Dict[OBVPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
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
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: Optional[bool] = True):
        if self._initialized and not force:
            return

        close = self._data["close"].to_numpy(np.float64)
        volume = self._data["volume"].to_numpy(np.float64)
        obv = np.zeros(close.size, dtype=np.float64)

        obv, obv_latest, close_latest = _obv(
            close=close,
            volume=volume,
            obv_container=obv,
        )

        if self._memory:
            self._obv = array("f", obv)

        self._obv_latest = obv_latest
        self._close_p = close_latest

        self.drop_data()
        self.set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series):
        close = data["close"]

        self._obv_latest = _obv_update(
            close=close,
            volume=data["volume"],
            close_p=self._close_p,
            obv_latest=self._obv_latest,
        )

        if self._memory:
            self._obv.append(self._obv_latest)

        self._close_p = close

        return self

    def fit(self, data, period_config: OBVPeriods = _period_config):
        return super().fit(data, period_config)

    def to_array(self, get: OBVKeys = "obv"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: OBVKeys = "obv",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: OBVKeys = "obv",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
