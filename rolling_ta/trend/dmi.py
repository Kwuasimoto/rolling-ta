from array import array
from typing import Literal, Optional
from rolling_ta.extras.numba import (
    _dm,
    _dm_update,
    _dm_smoothing,
    _dm_smoothing_update,
    _dmi,
    _dmi_update,
)
from rolling_ta.indicator import Indicator
from rolling_ta.volatility import TR
import pandas as pd
import numpy as np

from rolling_ta.logging import logger


class DMI(Indicator):
    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
        tr: Optional[TR] = None,
    ) -> None:
        super().__init__(data, period, memory, retention, init)
        self._n_1 = period - 1
        self._tr = TR(data, period, memory, retention, init) if tr is None else tr
        if self._init:
            self.init()

    def init(self):
        if not self._init:
            self._tr.init()

        high = self._data["high"].to_numpy(np.float64)
        low = self._data["low"].to_numpy(np.float64)
        tr = self.to_numpy("tr", np.float64)

        # pdm, ndm, pdm[-1], ndm[-1], high[-1], low[-1]

        pdm, ndm, high_p, low_p = _dm(
            high,
            low,
            np.zeros(high.size, dtype=np.float64),
            np.zeros(low.size, dtype=np.float64),
        )

        s_tr, self._s_tr_p = _dm_smoothing(
            tr, np.zeros(tr.size, dtype=np.float64), self._period_config
        )
        s_pdm, self._s_pdm_p = _dm_smoothing(
            pdm, np.zeros(pdm.size, dtype=np.float64), self._period_config
        )
        s_ndm, self._s_ndm_p = _dm_smoothing(
            ndm, np.zeros(ndm.size, dtype=np.float64), self._period_config
        )

        self._pdmi, self._pdmi_p = _dmi(
            s_pdm, s_tr, np.zeros(s_pdm.size, dtype=np.float64), self._period_config
        )
        self._ndmi, self._ndmi_p = _dmi(
            s_ndm, s_tr, np.zeros(s_ndm.size, dtype=np.float64), self._period_config
        )

        self._high_p = high_p
        self._low_p = low_p

        if self._memory:
            self._pdmi = array("d", self._pdmi)
            self._ndmi = array("d", self._ndmi)

        self.drop_data()
        self.set_initialized()

    def update(self, data: pd.Series):
        high = data["high"]
        low = data["low"]

        # Update sub indicators and get necessary values
        tr = self._tr.update(data)

        pdm, ndm = _dm_update(high, low, self._high_p, self._low_p)

        self._s_tr_p = _dm_smoothing_update(tr, self._s_tr_p, self._period_config)
        self._s_pdm_p = _dm_smoothing_update(pdm, self._s_pdm_p, self._period_config)
        self._s_ndm_p = _dm_smoothing_update(ndm, self._s_ndm_p, self._period_config)

        self._pdmi_p = _dmi_update(self._s_pdm_p, self._s_tr_p)
        self._ndmi_p = _dmi_update(self._s_ndm_p, self._s_tr_p)

        self._high_p = high
        self._low_p = low

        if self._memory:
            self._pdmi.append(self._pdmi_p)
            self._ndmi.append(self._ndmi_p)

    def to_array(self, get: Literal["pdmi", "ndmi", "tr"] = "pdmi"):
        if get == "tr":
            return self._tr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["pdmi", "ndmi", "tr"] = "pdmi",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        if get == "tr":
            return self._tr.to_numpy(get, dtype, **kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["pdmi", "ndmi", "tr"] = "pdmi",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        if get == "tr":
            self._tr.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)

    def pdmi_latest(self):
        return self._pdmi_p

    def ndmi_latest(self):
        return self._ndmi_p
