from pandas import DataFrame, Series
from rolling_ta.extras.numba import _dx, _adx
from rolling_ta.indicator import Indicator
from rolling_ta.volatility import TrueRange, NumbaTrueRange, AverageTrueRange
from rolling_ta.trend import DMI, NumbaDMI
from rolling_ta.logging import logger
import pandas as pd
import numpy as np

from typing import Optional, Union


def expMovingAverage(values, window):
    weights = np.exp(np.linspace(-1.0, 0.0, window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[: len(values)]
    a[:window] = a[window]
    return a


# Has 100% correct implementation.
class NumbaADX(Indicator):

    def __init__(
        self,
        data: DataFrame,
        period_config: int = 14,
        memory: bool = True,
        retention: Union[int, None] = None,
        init: bool = True,
        dmi: Optional[NumbaDMI] = None,
        tr: Optional[NumbaTrueRange] = None,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        self._dmi = (
            NumbaDMI(data, period_config, memory, retention, init, tr)
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

        self._dx = _dx(pdmi, ndmi, self._period_config)
        self._adx = _adx(self._dx, self._period_config, self._dmi._period_config)

        self.drop_data()

    def update(self, data: Series):
        raise NotImplementedError("ADX Update has yet to be implemented!")

    def adx(self):
        return self._adx


class ADX(Indicator):
    """
    A class to represent the Average Directional Index (ADX) indicator.

    The ADX is a technical indicator used to measure the strength of a trend.
    It is part of the Directional Movement System (DMS), which also includes
    the Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI).
    ADX helps identify strong trends and determine whether the market is trending or consolidating.

    Material
    --------
    - https://www.investopedia.com/terms/a/adx.asp
    - https://pypi.org/project/ta/

    Attributes
    ----------
    _adx : pd.Series
        A pandas Series storing the calculated ADX values.
    _adx_latest : float
        The most recent ADX value.
    _plus_di : pd.Series
        A pandas Series storing the Positive Directional Indicator (+DI) values.
    _minus_di : pd.Series
        A pandas Series storing the Negative Directional Indicator (-DI) values.
    _plus_di_latest : float
        The most recent +DI value.
    _minus_di_latest : float
        The most recent -DI value.
    _tr : pd.Series
        A pandas Series storing the True Range values, used in ADX calculation.
    _tr_latest : float
        The most recent True Range value.

    Methods
    -------
    **__init__(data: pd.DataFrame, period: int = 14, memory: bool = True, init: bool = True)** -> None

        Initializes the ADX indicator with the given data, period, and options.

    **init()** -> None

        Calculates the initial ADX, +DI, and -DI values based on the provided data.

    **update(data: pd.Series)** -> None

        Updates the ADX, +DI, and -DI based on new incoming data.

    **adx()** -> pd.Series

        Returns the stored ADX values if memory is enabled.

    **adx_latest()** -> float

        Returns the most recent ADX value.

    **plus_di()** -> pd.Series

        Returns the stored Positive Directional Indicator (+DI) values if memory is enabled.

    **plus_di_latest()** -> float

        Returns the most recent +DI value.

    **minus_di()** -> pd.Series

        Returns the stored Negative Directional Indicator (-DI) values if memory is enabled.

    **minus_di_latest()** -> float

        Returns the most recent -DI value.
    """

    _dmi: DMI

    _adx: pd.Series
    _adx_latest = np.nan

    def __init__(
        self,
        data: DataFrame,
        period: int = 14,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
        tr: Optional[TrueRange] = None,
        atr: Optional[AverageTrueRange] = None,
        dmi: Optional[DMI] = None,
    ) -> None:
        super().__init__(data, period, memory, retention, init)

        self._dmi = (
            DMI(data, period, memory, retention, init, tr, atr) if dmi is None else dmi
        )

        if self._init:
            self.init()

    def init(self):
        if not self._init:
            self._dmi.init()

        dmi = self._dmi.dmi()
        adx = dmi.ewm(
            span=self._period_config, min_periods=self._period_config, adjust=False
        ).mean()

        if self._memory:
            self._adx = adx
            self._count = adx.shape[0]

            if self._retention:
                self._adx = self.apply_retention(self._adx)

        self.drop_data()

    def update(self, data: Series):
        return super().update(data)

    def adx(self):
        return self._adx

    def dmi(self):
        return self._dmi.dmi()

    def pdi(self):
        return self._dmi.pdi()

    def ndi(self):
        return self._dmi.ndi()

    def atr(self):
        return self._dmi._atr.atr()

    def tr(self):
        return self._dmi._atr._tr.tr()
