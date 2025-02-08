from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _hma
from rolling_ta.trend.wma import WMA
from rolling_ta.indicator import Indicator


HMAKeys = Literal["hma", "wma_full", "wma_half"]
HMAPeriods = HMAKeys


class HMA(Indicator):

    _keys: List[HMAKeys] = ["hma", "wma_full", "wma_half"]
    _period_config: Dict[HMAPeriods, int] = {"hma": 14, "wma_full": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[HMAKeys] = _keys,
        period_config: Dict[HMAPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        wma_full: Optional[WMA] = None,
        wma_half: Optional[WMA] = None,
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
        if "wma_full" not in self._period_config:
            self._period_config.update({"wma_full": self._period_config["hma"]})
        if "wma_half" not in self._period_config:
            self._period_config.update(
                {"wma_half": self._period_config["wma_full"] // 2}
            )
        self._wma_full = (
            WMA(
                data,
                keys=["wma"],
                period_config={"wma": self._period_config["wma_full"]},
                memory=memory,
                retention=retention,
                init=init,
                force=force,
                initialization_state=initialization_state,
            )
            if wma_full is None
            else wma_full
        )
        self._wma_half = (
            WMA(
                data,
                keys=["wma"],
                period_config={"wma": self._period_config["wma_half"]},
                memory=memory,
                retention=retention,
                init=init,
                force=force,
                initialization_state=initialization_state,
            )
            if wma_half is None
            else wma_half
        )
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: bool = False):
        if self._initialized and not force:
            return

        if not self._wma_full._initialized:
            self._wma_full.calc(
                force=force,
                initialization_state=initialization_state,
            )
        if not self._wma_half._initialized:
            self._wma_half.calc(
                force=force,
                initialization_state=initialization_state,
            )

        close = self._data["close"].to_numpy(dtype=np.float64)
        wma_full = self._wma_full.to_numpy()
        wma_half = self._wma_half.to_numpy()
        hma_internim = np.zeros(close.size, dtype=np.float64)
        hma = np.zeros(close.size, dtype=np.float64)

        _hma(
            wma_full=wma_full,
            wma_half=wma_half,
            hma_internim=hma_internim,
            hma_container=hma,
            hma_period=self._period_config["hma"],
        )

        if self._memory:
            self._hma = array("d", hma)

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def fit(self, data, period_config: HMAPeriods = _period_config):
        super().fit(data, period_config)
        if "wma_full" in period_config:
            self._wma_full.fit(data, period_config={"wma": period_config["wma_full"]})
        if "wma_half" in period_config:
            self._wma_half.fit(data, period_config={"wma": period_config["wma_half"]})

    def to_array(self, get: HMAKeys = "hma"):
        if get == "wma_full":
            return self._wma_full.to_array("wma")
        elif get == "wma_half":
            return self._wma_half.to_array("wma")
        return super().to_array(get)

    def to_numpy(
        self,
        get: HMAKeys = "hma",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        if get == "wma_full":
            return self._wma_full.to_numpy("wma", dtype, **kwargs)
        elif get == "wma_half":
            return self._wma_half.to_numpy("wma", dtype, **kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: HMAKeys = "hma",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        if get == "wma_full":
            return self._wma_full.to_series("wma", dtype, name, **kwargs)
        elif get == "wma_half":
            return self._wma_half.to_series("wma", dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)
