from array import array
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import (
    _dm,
    _dm_update,
    _dm_smoothing,
    _dm_smoothing_update,
    _dmi,
    _dmi_update,
)
from rolling_ta.indicator import Indicator
from rolling_ta.volatility.tr import TrueRange, TrueRangeKeys

DMIKeys = Literal["pdmi", "ndmi", TrueRangeKeys]
DMIPeriods = DMIKeys


class DMI(Indicator):

    _keys: List[DMIKeys] = ["pdmi", "ndmi", "tr"]
    _period_config: Dict[DMIPeriods, int] = {
        "pdmi": 14,
        "ndmi": 14,
        "tr": 14,
    }

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[DMIKeys] = _keys,
        period_config: Dict[DMIPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        tr: Optional[TrueRange] = None,
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
        if "tr" not in self._period_config:
            self._period_config.update(
                {"tr": (self._period_config["pdmi"] + self._period_config["ndmi"]) // 2}
            )
        self._tr = (
            TrueRange(
                data,
                keys=["tr"],
                period_config={"tr": self._period_config["tr"]},
                memory=memory,
                retention=retention,
                init=init,
                force=force,
                initialization_state=initialization_state,
            )
            if tr is None
            else tr
        )
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: bool = False):
        if not self._init:
            self._tr.calc(
                force=force,
                initialization_state=initialization_state,
            )

        high = self._data["high"].to_numpy(np.float64)
        low = self._data["low"].to_numpy(np.float64)
        tr = self.to_numpy("tr", np.float64)

        pdm, ndm, high_p, low_p = _dm(
            high,
            low,
            np.zeros(high.size, dtype=np.float64),
            np.zeros(low.size, dtype=np.float64),
        )

        s_tr, self._s_tr_p = _dm_smoothing(
            tr, np.zeros(tr.size, dtype=np.float64), self._period_config["tr"]
        )
        s_pdm, self._s_pdm_p = _dm_smoothing(
            pdm, np.zeros(pdm.size, dtype=np.float64), self._period_config["pdmi"]
        )
        s_ndm, self._s_ndm_p = _dm_smoothing(
            ndm, np.zeros(ndm.size, dtype=np.float64), self._period_config["ndmi"]
        )

        self._pdmi, self._pdmi_p = _dmi(
            s_pdm,
            s_tr,
            np.zeros(s_pdm.size, dtype=np.float64),
            self._period_config["pdmi"],
        )
        self._ndmi, self._ndmi_p = _dmi(
            s_ndm,
            s_tr,
            np.zeros(s_ndm.size, dtype=np.float64),
            self._period_config["ndmi"],
        )

        self._high_p = high_p
        self._low_p = low_p

        if self._memory:
            self._pdmi = array("d", self._pdmi)
            self._ndmi = array("d", self._ndmi)

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series):
        high = data["high"]
        low = data["low"]

        # Update sub indicators and get necessary values
        tr = self._tr.update(data)

        pdm, ndm = _dm_update(high, low, self._high_p, self._low_p)

        self._s_tr_p = _dm_smoothing_update(tr, self._s_tr_p, self._period_config["tr"])
        self._s_pdm_p = _dm_smoothing_update(
            pdm, self._s_pdm_p, self._period_config["pdmi"]
        )
        self._s_ndm_p = _dm_smoothing_update(
            ndm, self._s_ndm_p, self._period_config["ndmi"]
        )

        self._pdmi_p = _dmi_update(self._s_pdm_p, self._s_tr_p)
        self._ndmi_p = _dmi_update(self._s_ndm_p, self._s_tr_p)

        self._high_p = high
        self._low_p = low

        if self._memory:
            self._pdmi.append(self._pdmi_p)
            self._ndmi.append(self._ndmi_p)

        return self

    def fit(
        self, data, period_config: Optional[Dict[DMIPeriods, int]] = _period_config
    ):
        super().fit(data, period_config)
        if set(period_config.keys()) & set(self._tr._period_config.keys()):
            self._tr.fit(data, period_config)

    def to_array(self, get: DMIKeys = "pdmi"):
        if get == "tr":
            return self._tr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: DMIKeys = "pdmi",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        if get == "tr":
            return self._tr.to_numpy(get, dtype, **kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: DMIKeys = "pdmi",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        if get == "tr":
            return self._tr.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)

    def pdmi_latest(self):
        return self._pdmi_p

    def ndmi_latest(self):
        return self._ndmi_p
