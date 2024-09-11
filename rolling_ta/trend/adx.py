from pandas import DataFrame, Series
from rolling_ta.indicator import Indicator

import pandas as pd
import numpy as np


class ADX(Indicator):
    """
    A class to represent the Average Directional Index (ADX) indicator.

    The ADX is a technical indicator used to measure the strength of a trend.
    It is part of the Directional Movement System (DMS), which also includes
    the Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI).
    ADX helps identify strong trends and determine whether the market is trending or consolidating.

    Material
    --------
        https://www.investopedia.com/terms/a/adx.asp
        https://pypi.org/project/ta/

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

    _adx: pd.Series
    _adx_latest = np.nan

    _plus_di: pd.Series
    _plus_di_latest = np.nan

    _neg_di: pd.Series
    _neg_di_latest = np.nan

    _true_range: pd.Series
    _true_range_latest = np.nan

    def __init__(self, data: DataFrame, period: int, memory: bool, init: bool) -> None:
        super().__init__(data, period, memory, init)

        if self._init:
            self.init()

    def init(self):
        return super().init()

    def update(self, data: Series):
        return super().update(data)
