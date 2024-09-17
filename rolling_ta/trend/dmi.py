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

        pdm, ndm, pdm_p, ndm_p = _dm(high, low)

        s_tr, s_tr_p = _dm_smoothing(tr)
        s_pdm, s_pdm_p = _dm_smoothing(pdm)
        s_ndm, s_ndm_p = _dm_smoothing(ndm)

        pdmi, pdmi_p = _dmi(s_pdm, s_tr, self._period_config)
        ndmi, ndmi_p = _dmi(s_ndm, s_tr, self._period_config)

        if self._memory:
            self._pdmi = pdmi
            self._ndmi = ndmi

        self._high_p = high[-1]
        self._low_p = low[-1]

        self._pdm_p = pdm_p
        self._ndm_p = ndm_p

        self._s_tr_p = s_tr_p
        self._s_pdm_p = s_pdm_p
        self._s_ndm_p = s_ndm_p

        self._pdmi_p = pdmi_p
        self._ndmi_p = ndmi_p

        self.drop_data()

    def update(self, data: pd.Series):
        high = data["high"]
        low = data["low"]

        tr_p = self._tr._tr_latest
        self._tr.update(data)

        pdm, ndm = _dm_update(high, low, self._high_p, self._low_p)

        s_tr = _dm_smoothing_update(self._tr._tr_latest, tr_p, self._period_config)
        s_pdm = _dm_smoothing_update(pdm, self._pdm_p, self._period_config)
        s_ndm = _dm_smoothing_update(ndm, self._ndm_p, self._period_config)

        self._pdmi_p = _dmi_update(s_pdm, s_tr)
        self._ndmi_p = _dmi_update(s_ndm, s_tr)

        self._pdm_p = pdm
        self._ndm_p = ndm

        self._s_tr_p = s_tr
        self._s_pdm_p = s_pdm
        self._s_ndm_p = s_ndm

        self._high_p = high
        self._low_p = low

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
