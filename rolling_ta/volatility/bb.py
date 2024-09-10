import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
from rolling_ta.trend import SMA


class BollingerBands(Indicator):

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
        """Rolling Bollinger Bands indicator

        https://www.investopedia.com/terms/s/sma.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=20 | Window length.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            weight (np.float64): Default=2.0 | The weight of the upper and lower bands.
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
        return self._sma.sma()

    def uband(self):
        return self._uband

    def lband(self):
        return self._lband

    def sma_latest(self):
        return self._sma._sma_latest

    def uband_latest(self):
        return self._uband_latest

    def lband_latest(self):
        return self._lband_latest
