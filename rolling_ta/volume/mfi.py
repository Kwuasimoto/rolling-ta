from pandas import DataFrame
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
import pandas as pd
import numpy as np

from collections import deque

from typing import Deque


class MFI(Indicator):
    """
    Money Flow Index (MFI) indicator.

    The MFI is a momentum indicator that uses both price and volume data to
    identify overbought or oversold conditions in an asset. This class calculates
    the MFI using historical price and volume data over a specified period.

    Material
    --------
        https://www.investopedia.com/terms/s/standarddeviation.asp
        https://pypi.org/project/ta/

    Attributes
    ----------
    _mfi : pd.Series
        A pandas Series storing the calculated MFI values for each period.
    _mfi_latest : float
        The latest MFI value.
    _prev_typical : float
        The typical price from the previous period (used for calculating up or down movement).
    _raw_mf : deque
        A deque storing raw money flow values for the current period.

    Methods
    -------
    **__init__(data: DataFrame, period: int = 14, memory: bool = True, init: bool = True)** -> None

        Initializes the MFI indicator with the given data, period, and options.

    **init()** -> None

        Calculates the initial MFI values based on the provided data.

    **update(data: pd.Series)** -> None

        Updates the MFI based on new incoming data.

    **mfi()** -> pd.Series

        Returns the stored MFI values if memory is enabled.

    **mfi_latest()** -> float

        Returns the latest MFI value.
    """

    _mfi: pd.Series
    _mfi_latest = np.nan

    _prev_typical = np.nan
    _raw_mf: deque

    def __init__(
        self,
        data: DataFrame,
        period: int = 14,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        """
        Initialize the MFI indicator.

        Args:
            data (pd.DataFrame): The initial dataframe containing price and volume data.
            period (int): Default=14 | The period over which to calculate the MFI.
            memory (bool): Default=True | Whether to store MFI values in memory.
            init (bool): Default=True | Whether to calculate the initial MFI values upon instantiation.
        """
        super().__init__(data, period, memory, init)

        if self._init:
            self.init()

    def init(self):
        """
        Calculate the initial MFI values based on historical data.
        """
        high = self._data["high"]
        low = self._data["low"]
        close = self._data["close"]
        volume = self._data["volume"]

        # Calculate typical and shift
        typical_price = (high + low + close) / 3

        # [1, 2, 3]
        # [nan, 1, 2]
        prev_typical = typical_price.shift(1)

        # Mark whether movement is up(1) down(-1) or neutral(0)
        up_down = np.where(
            typical_price > prev_typical,
            1,
            np.where(typical_price < prev_typical, -1, 0),
        )

        # Calculate raw money flow utilizing up_down.
        raw_mf = typical_price * volume * up_down

        # Summarize 14 period windows for both negative and positive raw money flows for each respectively.
        mfr_pos = raw_mf.rolling(self._period, min_periods=self._period).apply(
            lambda x: np.sum(np.where(x > 0, x, 0))
        )
        mfr_neg = raw_mf.rolling(self._period, min_periods=self._period).apply(
            lambda x: np.sum(np.where(x < 0, -x, 0))
        )

        # Calculate MFI.
        mfi = (100 * mfr_pos) / (mfr_pos + mfr_neg)
        self._mfi_latest = mfi.iloc[-1]

        if self._memory:
            self._mfi = mfi
            self._count = close.shape[0]

        # Save the money for for the last window summarization
        self._raw_mf = deque(raw_mf[-self._period :], maxlen=self._period)
        self._prev_typical = typical_price.iloc[-1]

        self._data = None

    def update(self, data: pd.Series):
        """
        Update the MFI with new incoming price and volume data.

        Args:
            data (pd.Series): The latest data containing 'high', 'low', 'close', and 'volume'.
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate typical price
        typical_price = (high + low + close) / 3

        # Get up_down marker
        up_down = (
            1
            if typical_price > self._prev_typical
            else -1 if typical_price < self._prev_typical else 0
        )

        # Calculate raw money flow and append to deque window
        raw_mf = typical_price * volume * up_down
        self._raw_mf.append(raw_mf)

        # Convert to pd.Series for vectorized calculations
        raw_mf = pd.Series(self._raw_mf)

        # Find pos and neg money flows
        mfr_pos = np.sum(np.where(raw_mf > 0, raw_mf, 0))
        mfr_neg = np.sum(np.where(raw_mf < 0, -raw_mf, 0))

        # Calculate MFI
        self._mfi_latest = (100 * mfr_pos) / (mfr_pos + mfr_neg)
        self._prev_typical = typical_price

        if self._memory:
            self._mfi[self._count] = self._mfi_latest
            self._count += 1

    def mfi(self):
        """
        Return the stored MFI values.

        Returns:
            pd.Series: The MFI values calculated over the historical data if memory is enabled.

        Raises:
            MemoryError: if function is called and memory = False
        """
        if not self._memory:
            raise MemoryError("MFI._memory = False")
        return self._mfi

    def mfi_latest(self):
        """
        Return the most recent MFI value.

        Returns:
            float: The latest MFI value.
        """
        return self._mfi_latest
