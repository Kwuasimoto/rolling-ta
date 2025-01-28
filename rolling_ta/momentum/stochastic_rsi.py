from array import array
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _stoch_k, _stoch_d
from rolling_ta.indicator import Indicator
from rolling_ta.momentum import RSI, RSI

from .rsi import RelativeStrengthIndexKeys

StochasticRSIKeys = Union[Literal["stoch_k", "stoch_d"], RelativeStrengthIndexKeys]
StochasticRSIPeriods = Union[Literal["stoch_rsi", StochasticRSIKeys]]


class StochasticRSI(Indicator):

    _keys: List[StochasticRSIKeys] = [
        "rsi",
        "stoch_rsi",
        "stoch_d",
    ]
    _period_default: Dict[StochasticRSIPeriods, int] = {
        "rsi": 14,
        "stoch_rsi": 10,
        "stoch_k": 3,
        "stoch_d": 3,
    }

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[StochasticRSIKeys] = _keys,
        period_config: Dict[StochasticRSIPeriods, int] = _period_default,
        memory: bool = True,
        retention: Optional[None] = None,
        init: bool = False,
        rsi: Optional[RSI] = None,
    ) -> None:
        super().__init__(
            data,
            keys=keys,
            period_config=period_config,
            memory=memory,
            retention=retention,
            init=init,
        )

        if "rsi" not in self._period_config:
            self._period_config.update({"rsi": 14})

        self._rsi = (
            RSI(
                data,
                keys=["rsi"],
                period_config={"rsi": self._period_config["rsi"]},
                memory=memory,
                retention=retention,
                init=init,
            )
            if rsi is None
            else rsi
        )

        if self._init:
            self.calc()

    def calc(self):
        if not self._rsi._initialized:
            self._rsi.calc()

        rsi = self._rsi.to_numpy()
        stoch_k = np.zeros(rsi.size, dtype=np.float64)

        self._window = _stoch_k(
            rsi,
            stoch_k,
            self._rsi._period_config["rsi"],
            self._period_config["stoch_rsi"],
            self._period_config["stoch_k"],
        )

        if self._period_config["stoch_d"] > 0:
            stoch_d = np.array(stoch_k, dtype=np.float64)
            _stoch_d(
                stoch_k,
                stoch_d,
                self._rsi._period_config["rsi"],
                self._period_config["stoch_rsi"],
                self._period_config["stoch_d"],
            )

        if self._memory:
            self._stoch_rsi = array("d", stoch_k)

            if stoch_d is not None:
                self._stoch_d = array("d", stoch_d)

        return self

    def update(self, data: pd.Series):
        return super().update(data, __name__)

    def fit(
        self,
        data,
        period_config: Optional[Dict[StochasticRSIPeriods, int]] = _period_default,
    ):
        super().fit(data, period_config)
        if set(period_config.keys()) & set(self._rsi._period_config.keys()):
            self._rsi.fit(data, period_config)

    def to_array(self, get: StochasticRSIKeys = "stoch_k"):
        if get == "rsi":
            return self._rsi.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: StochasticRSIKeys = "stoch_k",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        if get == "rsi":
            return self._rsi.to_numpy(get, dtype, **kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: StochasticRSIKeys = "stoch_k",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        if get == "rsi":
            return self._rsi.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)
