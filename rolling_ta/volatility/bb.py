import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
from rolling_ta.trend import SMA


class BollingerBands(Indicator):

    _sma: SMA

    _weight = np.nan

    _latest_uband = np.nan
    _latest_lband = np.nan

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

        self._sma = SMA(self._data, self._period, False, False)
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

        self._latest_uband = uband.iloc[-1]
        self._latest_lband = lband.iloc[-1]

        # Calculate initital BB Values
        self.calculate()

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
        self.calculate()

        if self._memory:
            self._uband[self._count] = self._latest_uband
            self._lband[self._count] = self._latest_lband
            self._count += 1

    def calculate(self):
        std = np.std(self._sma._window, ddof=0)
        std_weighted = std * self._weight

        self._latest_uband = self._sma._latest_sma + std_weighted
        self._latest_lband = self._sma._latest_sma - std_weighted

    def latest_sma(self):
        return self._sma._latest_sma

    def latest_uband(self):
        return self._latest_uband

    def latest_lband(self):
        return self._latest_lband
