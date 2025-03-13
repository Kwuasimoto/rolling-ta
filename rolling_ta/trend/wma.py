from array import array
from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _wma, _wma_update
from rolling_ta.indicator import Indicator


WMAKeys = Literal["wma"]
WMAPeriods = Literal["wma"]


class WMA(Indicator):

    _keys: List[WMAKeys] = ["wma"]
    _period_config: Dict[WMAPeriods, int] = {"wma": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[WMAKeys] = _keys,
        period_config: Dict[WMAPeriods, int] = _period_config,
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

        close = self._data["close"].to_numpy(dtype=np.float64)
        wma = np.zeros(close.size, dtype=np.float64)

        _wma(
            close=close,
            wma_container=wma,
            period=self._period_config["wma"],
        )

        self._price_latest = close[-self._period_config["wma"] :]

        if self._memory:
            self._wma = array("d", wma)

        self.drop_data()
        self.set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series) -> Indicator:
        close = data["close"]

        wma = _wma_update(
            price=close,
            price_latest=self._price_latest,
            period=self._period_config["wma"],
        )

        if self._memory:
            self._wma.append(wma)

        return self

    def get(self, index: int, key: WMAKeys = "wma"):
        return super().get(index, key)

    def fit(
        self,
        data: pd.DataFrame,
        period_config: WMAPeriods = _period_config,
    ):
        return super().fit(data, period_config)

    def to_array(self, get: WMAKeys = "wma"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: WMAKeys = "wma",
        dtype: Optional[None] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: WMAKeys = "wma",
        dtype: Optional[None] = float,
        name: Optional[None] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
