from typing import Dict, List, Literal, Optional, Union
from array import array

import pandas as pd
import numpy as np

from rolling_ta.extras.numba import _dx, _adx, _dx_update, _adx_update
from rolling_ta.indicator import Indicator
from rolling_ta.volatility import TrueRange
from rolling_ta.trend import DMI, DMI

from .dmi import DMIKeys

ADXKeys = Union[Literal["adx", "dx", DMIKeys]]
ADXPeriods = ADXKeys


class ADX(Indicator):

    _keys: ADXKeys = ["adx", "dx", "dmi", "pdmi", "ndmi", "tr"]
    _period_config: Dict[ADXPeriods, int] = {
        "adx": 14,
        "dx": 14,
        "pdmi": 14,
        "ndmi": 14,
        "tr": 14,
    }

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[ADXKeys] = ["adx", "dx", "dmi", "pdmi", "ndmi", "tr"],
        period_config: Dict[ADXPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        dmi: Optional[DMI] = None,
        tr: Optional[TrueRange] = None,
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

        if "dx" not in self._period_config:
            self._period_config.update({"dx": self._period_config["adx"]})

        if "pdmi" not in self._period_config:
            self._period_config.update({"pdmi": self._period_config["dx"]})

        if "ndmi" not in self._period_config:
            self._period_config.update({"ndmi": self._period_config["dx"]})

        if "tr" not in self._period_config:
            self._period_config.update({"tr": self._period_config["dx"]})

        self._dmi = (
            DMI(
                data,
                keys=["pdmi", "ndmi", "tr"],
                period_config={
                    "pdmi": self._period_config["pdmi"],
                    "ndmi": self._period_config["ndmi"],
                    "tr": self._period_config["tr"],
                },
                memory=memory,
                retention=retention,
                init=init,
                tr=tr,
                force=force,
                initialization_state=initialization_state,
            )
            if dmi is None
            else dmi
        )
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: Optional[bool] = True):
        if self._initialized and not force:
            return

        if not self._dmi._initialized or force:
            self._dmi.calc(
                force=force,
                initialization_state=initialization_state,
            )

        pdmi = self.to_numpy(get="pdmi", dtype=np.float64)
        ndmi = self.to_numpy(get="ndmi", dtype=np.float64)

        dx, dx_p = _dx(
            pdmi,
            ndmi,
            np.zeros(pdmi.size, dtype=np.float64),
            self._period_config["dx"],
        )

        adx, adx_p = _adx(
            dx,
            np.zeros(dx.size, dtype=np.float64),
            self._period_config["adx"],
            self._period_config.get(
                "dmi", (self._period_config["pdmi"] + self._period_config["ndmi"]) // 2
            ),
        )

        if self._memory:
            self._adx = array("f", adx)
            self._dx = array("f", dx)

        self._dx_p = dx_p
        self._adx_p = adx_p

        self.drop_data()
        self._set_initialized(state=initialization_state)

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
            self._period_config["adx"],
        )

        if self._memory:
            self._adx.append(self._adx_p)

        return self

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Optional[Dict[ADXPeriods, int]] = _period_config,
    ):
        super().fit(data, period_config)
        if set(period_config.keys()) & set(self._dmi._period_config.keys()):
            self._dmi.fit(data, period_config)

    def to_array(self, get: ADXKeys = "adx"):
        if get == "pdmi":
            return self._dmi.to_array(get)
        elif get == "ndmi":
            return self._dmi.to_array(get)
        elif get == "tr":
            return self._dmi._tr.to_array(get)
        return super().to_array(get)

    def to_numpy(
        self,
        get: ADXKeys = "adx",
        dtype: Optional[np.dtype] = np.float64,
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
        get: ADXKeys = "adx",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
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
