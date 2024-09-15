from typing import Optional
from rolling_ta.extras.numba import _dm, _dm_smoothing, _dmi, _dx
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

        pdm, ndm = _dm(high, low)
        self._pdm = pdm
        self._ndm = ndm

        # ADX Does not use the initial TR value (high-low). Only valid True Range calculations.
        self._s_tr = _dm_smoothing(tr)
        self._s_pdm = _dm_smoothing(pdm)
        self._s_ndm = _dm_smoothing(ndm)

        self._pdmi = _dmi(self._s_pdm, self._s_tr)
        self._ndmi = _dmi(self._s_ndm, self._s_tr)

        self.drop_data()

    def update(self, data: pd.Series):
        return super().update(data)

    def pdmi(self):
        return pd.Series(self._pdmi)

    def ndmi(self):
        return pd.Series(self._ndmi)


class DMI(Indicator):

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

            # if self._retention:
            #     self.apply_retention()

        self.drop_data()

    def update(self, data: pd.Series):
        return super().update(data)

    def pdi(self):
        return self._pdi

    def ndi(self):
        return self._ndi

    def dmi(self):
        return self._dmi
