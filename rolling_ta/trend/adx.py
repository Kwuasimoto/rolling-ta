from typing import Literal, Optional
from array import array

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _dx, _adx, _dx_update, _adx_update
from rolling_ta.indicator import Indicator
from rolling_ta.volatility import TrueRange
from rolling_ta.trend import DMI, DMI
from rolling_ta.logging import log


class ADX(Indicator):

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: int = 14,
        memory: bool = True,
        retention: Optional[int] = None,
        columns: Optional[list[str]] = None,
        init: bool = False,
        dmi: Optional[DMI] = None,
        tr: Optional[TrueRange] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, columns, init)
        self._n_1 = period_config - 1
        self._dmi = (
            DMI(data, period_config, memory, retention, columns, init, tr)
            if dmi is None
            else dmi
        )
        if self._init:
            self.set_columns(columns)
            self.calc()

    def calc(self):
        if not self._dmi._initialized:
            self._dmi.calc()

        pdmi = self.to_numpy(get="pdmi", dtype=np.float64)
        ndmi = self.to_numpy(get="ndmi", dtype=np.float64)

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
            self._dx = array("f", dx)

        if self._columns is None:
            self.set_columns()

        self._dx_p = dx_p
        self._adx_p = adx_p

        self.drop_data()
        self.set_initialized()

        return self

    def update(self, data: pd.Series):
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

        return self

    def fit(self, data, period_config: Optional[int] = None):
        super().fit(data, period_config)
        self._dmi.fit(data, period_config)

    def set_columns(self, columns=None, name=None):
        super().set_columns(
            f"adx_{self._period_config}" if columns is None else columns, name
        )
        self._dmi.set_columns(
            [f"pdmi_{self._dmi._period_config}", f"ndmi_{self._dmi._period_config}"]
        )

    def to_array(self, get: Literal["adx", "dx", "pdmi", "ndmi", "tr"] = "adx"):
        if get == "pdmi":
            return self._dmi.to_array(get)
        elif get == "ndmi":
            return self._dmi.to_array(get)
        elif get == "tr":
            return self._dmi._tr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: Literal["adx", "dx", "pdmi", "ndmi", "tr"] = "adx",
        dtype: np.dtype | None = np.float64,
        **kwargs,
    ):
        if get == "pdmi":
            return self._dmi.to_numpy(get, dtype, **kwargs)
        elif get == "ndmi":
            return self._dmi.to_numpy(get, dtype, **kwargs)
        elif get == "tr":
            return self._dmi._tr.to_numpy(get, dtype, **kwargs)
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: Literal["adx", "dx", "pdmi", "ndmi", "tr"] = "adx",
        dtype: type | None = float,
        name: str | None = None,
        **kwargs,
    ):
        if get == "pdmi":
            return self._dmi.to_series(get, dtype, name, **kwargs)
        elif get == "ndmi":
            return self._dmi.to_series(get, dtype, name, **kwargs)
        elif get == "tr":
            return self._dmi._tr.to_series(get, dtype, name, **kwargs)
        return super().to_series(get, dtype, name, **kwargs)

    def adx_latest(self):
        return self._adx_p

    def dx_latest(self):
        return self._dx_p
