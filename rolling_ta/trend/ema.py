from array import array
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _ema, _ema_update
from rolling_ta.indicator import Indicator

EMAKeys = Literal["ema"]
EMAPeriods = Literal["ema", "weight"]


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) Indicator.

    The EMA gives more weight to recent prices, making it more responsive to new information compared to the Simple Moving Average (SMA).
    This indicator is commonly used to identify trends and smooth out price data.

    Material
    --------
        https://www.investopedia.com/terms/e/ema.asp
    """

    _keys: List[EMAKeys] = ["ema", "weight"]
    _period_config: Dict[EMAPeriods, int] = {"ema": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[EMAKeys] = _keys,
        period_config: Dict[EMAPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        weight: np.float64 = 2.0,
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
        self._weight = (
            weight / (period_config["ema"] + 1)
            if "weight" not in period_config
            else period_config["weight"]
        )
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def get(self, index: int, key: EMAKeys = "ema"):
        return super().get(index, key)

    def calc(self, force: bool = False, initialization_state: bool = False):
        if self._initialized and not force:
            return

        close = self._data["close"].to_numpy(dtype=np.float64)
        ema = np.zeros(close.size)

        ema, ema_latest = _ema(
            close,
            ema,
            self._weight,
            self._period_config["ema"],
        )

        self._ema_latest = ema_latest

        if self._memory:
            self._ema = array("d", ema)

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series):
        self._ema_latest = _ema_update(data["close"], self._weight, self._ema_latest)

        if self._memory:
            self._ema.append(self._ema_latest)

        return self

    def fit(self, data, period_config=_period_config):
        return super().fit(data, period_config)

    def to_array(self, get: EMAKeys = "ema"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: EMAKeys = "ema",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: EMAKeys = "ema",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
