import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
from rolling_ta.trend import SMA


class BollingerBands(Indicator):
    """
    Bollinger Bands Indicator.

    Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations) which are used to
    identify volatility and potential overbought or oversold conditions in an asset.

    Material
    --------
        https://www.investopedia.com/terms/b/bollingerbands.asp
        https://pypi.org/project/ta/

    Attributes
    ----------
    _sma : SMA
        The Simple Moving Average (SMA) used to calculate the middle band.
    _weight : float
        The weight for the upper and lower bands, typically 2.0.
    _uband_latest : float
        The latest upper Bollinger Band value.
    _lband_latest : float
        The latest lower Bollinger Band value.
    _uband : pd.Series
        A pandas Series storing the upper Bollinger Band values.
    _lband : pd.Series
        A pandas Series storing the lower Bollinger Band values.

    Methods
    -------
    **__init__(data: pd.DataFrame, period: int = 20, weight: np.float16 = 2.0, memory: bool = True, init: bool = True)** -> None

        Initializes the Bollinger Bands indicator with the given data, period, weight, and options.

    **init()** -> None

        Calculates the initial Bollinger Bands values based on the provided data.

    **update(close: np.float64)** -> None

        Updates the Bollinger Bands based on new incoming closing price data.

    **sma()** -> pd.Series

        Returns the stored Simple Moving Average (SMA) values if memory is enabled.

    **uband()** -> pd.Series

        Returns the stored upper Bollinger Band values if memory is enabled.

    **lband()** -> pd.Series

        Returns the stored lower Bollinger Band values if memory is enabled.

    **sma_latest()** -> float

        Returns the latest SMA value.

    **uband_latest()** -> float

        Returns the latest upper Bollinger Band value.

    **lband_latest()** -> float

        Returns the latest lower Bollinger Band value.
    """

    _sma: SMA

    _weight = np.nan

    _uband_latest = np.nan
    _lband_latest = np.nan

    _uband: pd.Series
    _lband: pd.Series

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 20,
        weight: np.float16 = 2.0,
        memory: bool = True,
        init: bool = True,
    ) -> None:
        """
        Initializes the Bollinger Bands indicator with the given data, period, weight, and options.

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=20 | Window length for the SMA and standard deviation calculations.
            weight (np.float16): Default=2.0 | The weight of the upper and lower bands.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
        """
        super().__init__(data, period, memory, init)
        logger.debug(
            f"BollingerBands init: [data_len={len(data)}, period={period}, memory={memory}, init={init}]"
        )

        self._sma = SMA(self._data, self._period, self._memory, False)
        self._weight = weight

        if init:
            self.init()

    def init(self):
        """
        Calculates the initial Bollinger Bands values based on the provided data.

        This method computes the upper and lower Bollinger Bands using the SMA and standard deviation of the
        closing prices. It also initializes internal attributes and stores the computed Bollinger Bands values
        if memory is enabled.
        """
        self._sma.init()

        close = self._data["close"]
        count = close.shape[0]

        std = close.rolling(self._period, min_periods=self._period).std(ddof=0)
        sma = self._sma.sma()

        std_weighted = std * self._weight
        uband = sma + std_weighted
        lband = sma - std_weighted

        self._uband_latest = uband.iloc[-1]
        self._lband_latest = lband.iloc[-1]

        # Calculate initital BB Values
        std = np.std(self._sma._window, ddof=0)
        std_weighted = std * self._weight

        self._uband_latest = self._sma._sma_latest + std_weighted
        self._lband_latest = self._sma._sma_latest - std_weighted

        if self._memory:
            self._count = count
            self._uband = uband
            self._lband = lband
        else:
            self._data = None
            self._sma._data = None

    def update(self, close: np.float64):
        """
        Updates the Bollinger Bands based on new incoming closing price data.

        Args:
            close (np.float64): The new closing price to update the indicator with.
        """
        # Update SMA
        self._sma.update(close)

        # Calculate initital BB Values
        std = np.std(self._sma._window, ddof=0)
        std_weighted = std * self._weight

        self._uband_latest = self._sma._sma_latest + std_weighted
        self._lband_latest = self._sma._sma_latest - std_weighted

        if self._memory:
            self._uband[self._count] = self._uband_latest
            self._lband[self._count] = self._lband_latest
            self._count += 1

    def sma(self):
        """
        Returns the stored SMA values.

        Returns:
            pd.Series: A pandas Series containing the SMA values.
        """
        return self._sma.sma()

    def uband(self):
        """
        Returns the stored upper Bollinger Band values.

        Returns:
            pd.Series: A pandas Series containing the upper Bollinger Band values.
        """
        return self._uband

    def lband(self):
        """
        Returns the stored lower Bollinger Band values.

        Returns:
            pd.Series: A pandas Series containing the lower Bollinger Band values.
        """
        return self._lband

    def sma_latest(self):
        """
        Returns the latest SMA value.

        Returns:
            float: The most recent SMA value.
        """
        return self._sma._sma_latest

    def uband_latest(self):
        """
        Returns the latest upper Bollinger Band value.

        Returns:
            float: The most recent upper Bollinger Band value.
        """
        return self._uband_latest

    def lband_latest(self):
        """
        Returns the latest lower Bollinger Band value.

        Returns:
            float: The most recent lower Bollinger Band value.
        """
        return self._lband_latest
