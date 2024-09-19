from array import array
from typing import Optional
from rolling_ta.extras.numba import (
    _dm,
    _dm_update,
    _dm_smoothing,
    _dm_smoothing_update,
    _dmi,
    _dmi_update,
)
from rolling_ta.indicator import Indicator
from rolling_ta.volatility import TrueRange, NumbaTrueRange, AverageTrueRange
import pandas as pd
import numpy as np

from rolling_ta.logging import logger


class NumbaDMI(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
        tr: Optional[NumbaTrueRange] = None,
    ) -> None:
        super().__init__(data, period, memory, retention, init)
        self._n_1 = period - 1
        self._tr = (
            NumbaTrueRange(data, period, memory, retention, init) if tr is None else tr
        )
        if self._init:
            self.init()

    def init(self):
        if not self._init:
            self._tr.init()

        high = self._data["high"].to_numpy(np.float64)
        low = self._data["low"].to_numpy(np.float64)
        tr = self._tr.tr().to_numpy(np.float64)

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

    def pdmi(self):
        return pd.Series(self._pdmi)

    def ndmi(self):
        return pd.Series(self._ndmi)

    def pdmi_latest(self):
        return self._pdmi_p

    def ndmi_latest(self):
        return self._ndmi_p


# Incorrect implementation, use Numba version.
class DMI(Indicator):
    """_summary_

    Deprecated: Incorrect implementation, use NumbaADX / NumbaDMI.
    """

    _pdi: pd.Series
    _ndi: pd.Series
    _dmi: pd.Series

    _pdi_latest = np.nan
    _ndi_latest = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
        tr: Optional[TrueRange] = None,
        atr: Optional[AverageTrueRange] = None,
    ) -> None:
        super().__init__(data, period, memory, retention, init)

        self._atr = (
            AverageTrueRange(data, period, memory, retention, init, tr)
            if atr is None
            else atr
        )

        if self._init:
            self.init()

    def init(self):
        if not self._init:
            self._atr.init()

        high = self._data["high"]
        low = self._data["low"]

        high_p = high.shift(1)
        low_p = low.shift(1)

        move_up = high - high_p
        move_down = low_p - low

        pos_mask = (move_up > move_down) & (move_up > 0)
        neg_mask = (move_down > move_up) & (move_down > 0)

        pdm = np.zeros_like(move_up)
        ndm = np.zeros_like(move_down)

        pdm[pos_mask] = move_up[pos_mask]
        ndm[neg_mask] = move_down[neg_mask]

        pdm_emw = (
            pd.Series(pdm)
            .ewm(
                span=self._period_config, min_periods=self._period_config, adjust=False
            )
            .mean()
        )
        ndm_emw = (
            pd.Series(ndm)
            .ewm(
                span=self._period_config, min_periods=self._period_config, adjust=False
            )
            .mean()
        )

        atr = self._atr.atr()

        pdi = 100 * (pdm_emw / atr)
        ndi = 100 * (ndm_emw / atr)
        dmi = 100 * ((pdi - ndi).abs() / (pdi + ndi))

        if self._memory:
            self._pdi = pdi
            self._ndi = ndi
            self._dmi = dmi
            self._count = dmi.shape[0]

        self.drop_data()

    def update(self, data: pd.Series):
        raise NotImplementedError(
            "DMI Update not implemented. Please create a PR to help :D"
        )

    def pdi(self):
        return self._pdi

    def ndi(self):
        return self._ndi

    def dmi(self):
        return self._dmi
