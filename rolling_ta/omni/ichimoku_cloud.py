from array import array
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import (
    _kijun,
    _kijun_update,
    _senkou_a,
    _senkou_a_update,
    _senkou_b,
    _senkou_b_update,
    _tenkan,
    _tenkan_update,
)
from rolling_ta.indicator import Indicator

IchimokuCloudKeys = Literal["tenkan", "kijun", "senkou_a", "senkou_b"]
IchiMokuCloudPeriods = IchimokuCloudKeys


class IchimokuCloud(Indicator):
    """
    Ichimoku Cloud indicator, a comprehensive tool used in technical analysis
    to assess trends, support/resistance levels, and momentum.

    The Ichimoku Cloud calculates five key components:
    1. Tenkan-sen (Conversion Line)
    2. Kijun-sen (Base Line)
    3. Senkou Span A (Leading Span A)
    4. Senkou Span B (Leading Span B)
    5. Chikou Span (Lagging Span).

    Required Dictionary Keys:
    -------------------------
    The `periods` dictionary passed to the class must contain the following keys:

    - 'tenkan': (int) Period for the Tenkan-sen (conversion line), typically 9.
    - 'kijun': (int) Period for the Kijun-sen (base line), typically 26.
    - 'senkou': (int) Period for the Senkou Span B (leading span B), typically 52.

    Optional Dictionary Key:
    ------------------------
    - 'lagging': (int, optional) Custom period for the Chikou Span (lagging span).
      If not provided, it defaults to the 'kijun' period (typically 26).
    """

    _keys: List[IchimokuCloudKeys] = ["tenkan", "kijun", "senkou_a", "senkou_b"]
    _period_default: Dict[IchiMokuCloudPeriods, int] = {
        "tenkan": 9,
        "kijun": 26,
        "senkou_b": 52,
    }

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[IchimokuCloudKeys] = _keys,
        period_config: Dict[IchiMokuCloudPeriods, int] = _period_default,
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

        # Senkou_a should not be supplied, but its possible for weird indicator configurations.
        if "senkou_a" not in period_config:
            period_config["senkou_a"] = period_config["kijun"] + period_config["tenkan"]

        # Deconstruct period config to attributes to avoid function overhead
        self._clip = max(
            self._period_config["tenkan"],
            self._period_config["kijun"],
            self._period_config["senkou_a"],
            self._period_config["senkou_b"],
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

        tenkan = np.zeros(high.size, dtype=np.float64)
        _tenkan(high, low, tenkan, self._period_config["tenkan"])

        kijun = np.zeros(high.size, dtype=np.float64)
        _kijun(high, low, kijun, self._period_config["kijun"])

        senkou_a = np.zeros(high.size, dtype=np.float64)
        _senkou_a(
            tenkan,
            kijun,
            senkou_a,
            self._period_config["tenkan"],
            self._period_config["kijun"],
        )

        senkou_b = np.zeros(high.size, dtype=np.float64)
        _senkou_b(high, low, senkou_b, self._period_config["senkou_b"])

        if self._memory:
            self._tenkan = array("d", tenkan)
            self._kijun = array("d", kijun)
            self._senkou_a = array("d", senkou_a)
            self._senkou_b = array("d", senkou_b)

        self._tenkan_latest = tenkan[-1]
        self._kijun_latest = kijun[-1]
        self._senkou_a_latest = senkou_a[-1]
        self._senkou_b_latest = senkou_b[-1]

        self._high = high[-self._clip :]
        self._low = low[-self._clip :]

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series):
        self._high = np.roll(self._high, -1)
        self._high[-1] = data["high"]

        self._low = np.roll(self._low, -1)
        self._low[-1] = data["low"]

        tenkan = _tenkan_update(self._high, self._low, self._period_config["tenkan"])
        kijun = _kijun_update(self._high, self._low, self._period_config["kijun"])

        senkou_b = _senkou_b_update(
            self._high,
            self._low,
            self._period_config["senkou_b"],
        )

        senkou_a = _senkou_a_update(tenkan, kijun)

        if self._memory:
            self._tenkan.append(tenkan)
            self._kijun.append(kijun)
            self._senkou_a.append(senkou_a)
            self._senkou_b.append(senkou_b)

        self._tenkan_latest = tenkan
        self._kijun_latest = kijun
        self._senkou_a_latest = senkou_a
        self._senkou_b_latest = senkou_b

        return self

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Dict[IchiMokuCloudPeriods, int] = _period_default,
    ):
        return super().fit(data, period_config)

    def to_array(self, get: IchimokuCloudKeys = "tenkan"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: IchimokuCloudKeys = "tenkan",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: IchimokuCloudKeys = "tenkan",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)

    def tenkan_latest(self):
        return self._tenkan_latest

    def kijun_latest(self):
        return self._kijun_latest

    def senkou_a_latest(self):
        return self._senkou_a_latest

    def senkou_b_latest(self):
        return self._senkou_b_latest
