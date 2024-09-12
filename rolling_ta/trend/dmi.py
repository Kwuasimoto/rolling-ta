from typing import Optional
from rolling_ta.indicator import Indicator
from rolling_ta.volatility import TrueRange, AverageTrueRange
import pandas as pd
import numpy as np

from time import time

from rolling_ta.logging import logger


class DirectionalMovementIndex(Indicator):

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
