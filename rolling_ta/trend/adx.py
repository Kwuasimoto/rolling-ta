from array import array
from pandas import DataFrame, Series
from rolling_ta.extras.numba import _dx, _adx, _dx_update, _adx_update
from rolling_ta.indicator import Indicator
from rolling_ta.volatility import TR
from rolling_ta.trend import DMI, DMI
import pandas as pd
import numpy as np

from typing import Optional, Union


class ADX(Indicator):

    def __init__(
        self,
        data: DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: Union[int, None] = None,
        init: bool = True,
        dmi: Optional[DMI] = None,
        tr: Optional[TR] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        self._n_1 = period_config - 1
        self._dmi = (
            DMI(data, period_config, memory, retention, init, tr)
            if dmi is None
            else dmi
        )
        if self._init:
            self.init()

    def init(self):
        if not self._init:
            self._dmi.init()

        pdmi = self._dmi.pdmi().to_numpy(np.float64)
        ndmi = self._dmi.ndmi().to_numpy(np.float64)

        dx, dx_p = _dx(
            pdmi,
            ndmi,
            np.zeros(pdmi.size, dtype=np.float64),
            self._period_config,
        )

        adx, adx_p = _adx(
            dx,
            np.zeros(dx.size, dtype=np.float64),
            self._period_config,
            self._dmi._period_config,
        )

        if self._memory:
            self._adx = array("f", adx)

        self._dx_p = dx_p
        self._adx_p = adx_p

        self.drop_data()

    def update(self, data: Series):
        self._dmi.update(data)

        self._dx_p = _dx_update(
            self._dmi.pdmi_latest(),
            self._dmi.ndmi_latest(),
        )
        self._adx_p = _adx_update(
            self._dx_p,
            self._adx_p,
            self._period_config,
            self._n_1,
        )

        if self._memory:
            self._adx.append(self._adx_p)

    def adx(self):
        return pd.Series(self._adx)

    def adx_latest(self):
        return self._adx_p

    def dx_latest(self):
        return self._dx_p
