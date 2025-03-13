from array import array
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _bop, _bop_update
from rolling_ta.indicator import Indicator

BalanceOfPowerPeriods = Literal["bop"]
BalanceOfPowerKeys = Literal["bop"]


class BOP(Indicator):
    """Balance of Power"""

    _keys: List[BalanceOfPowerKeys] = ["bop"]
    _period_config: Dict[BalanceOfPowerPeriods, int] = {"bop": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[BalanceOfPowerKeys] = _keys,
        period_config: Dict[BalanceOfPowerPeriods, int] = _period_config,
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

    def calc(self, force: bool = False, initialization_state: bool = True):
        if self._initialized and not force:
            return

        open = self._data["open"].to_numpy(dtype=np.float64)
        high = self._data["high"].to_numpy(dtype=np.float64)
        low = self._data["low"].to_numpy(dtype=np.float64)
        close = self._data["close"].to_numpy(dtype=np.float64)

        bop = np.zeros(close.size, dtype=np.float64)
        self._latest_range = np.zeros(self._period_config["bop"] or 1, dtype=np.float64)

        _bop(
            open=open,
            high=high,
            low=low,
            close=close,
            bop_container=bop,
            latest_range_container=self._latest_range,
            smoothing=self._period_config["bop"],
        )

        if self._memory:
            self._bop = array("d", bop)

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series) -> Indicator:
        self._bop_latest = _bop_update(
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            latest_range=self._latest_range,
            smoothing=self._period_config["bop"],
        )

        if self._memory:
            self._bop.append(self._bop_latest)

        return self

    def fit(
        self,
        data,
        period_config: Optional[Dict[BalanceOfPowerPeriods, int]] = None,
    ):
        if period_config is None:
            super().fit(data, self._period_config)
            return
        super().fit(data, period_config)

    def to_array(self, get: BalanceOfPowerKeys = "bop"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: BalanceOfPowerKeys = "bop",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: BalanceOfPowerKeys = "bop",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ) -> pd.Series:
        return super().to_series(get, dtype, name, **kwargs)
