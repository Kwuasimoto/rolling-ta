import numpy as np
from pandas import DataFrame

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
from rolling_ta.trend import SMA


class BollingerBands(Indicator):

    _sma: SMA

    _upper_band = np.nan
    _lower_band = np.nan

    _weight = np.nan

    def __init__(
        self,
        data: DataFrame,
        period: int = 20,
        weight: np.float64 = 2.0,
        memory: bool = True,
        init: bool = True,
        roll: bool = True,
    ) -> None:
        """Rolling Bollinger Bands indicator

        https://www.investopedia.com/terms/s/sma.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=20 | Window length.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            weight (np.float64): Default=2.0 | The weight of the upper and lower bands.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
            roll (bool, optional): Default=True | Calculate remaining indicator values upon instantiation.
        """
        super().__init__(data, period, memory, init, roll)
        logger.debug(
            f"BollingerBands init: [data_len={len(data)}, period={period}, memory={memory}, init={init}]"
        )
        self._weight = weight

        if init:
            self.init()

    def init(self):
        # We want the SMA to roll with the BollingerBands, not independently.
        close = self._data["close"]
        count = close.shape[0]

        self._sma = SMA(self._data, self._period, False, self._init, False)

        # Calculate initital BB Values
        self.calculate()

        if self._memory:
            self._count = count
            self._data[f"bb_mband_{self._period}"] = np.nan
            self._data[f"bb_uband_{self._period}"] = np.nan
            self._data[f"bb_lband_{self._period}"] = np.nan
            self.save(self._period - 1)
        else:
            self._data = None
            self._sma._data = None

        # Roll the rest of the BB
        if self._roll:
            for i in range(self._period, count):
                current_close = close[i]
                self._sma.update(current_close)
                self.update(current_close)

    def calculate(self):
        self._stddev = np.std(self._sma._window)
        stddev_weighted = self._stddev * self._weight

        self._upper_band = self._sma._latest_value + stddev_weighted
        self._lower_band = self._sma._latest_value - stddev_weighted

    def save(self, index: int):
        self._data.at[index, f"bb_mband_{self._period}"] = self._sma._latest_value
        self._data.at[index, f"bb_uband_{self._period}"] = self._upper_band
        self._data.at[index, f"bb_lband_{self._period}"] = self._lower_band

    def update(self, close: np.float64):
        self._sma.update(close)

        # Calculate initital BB Values
        self.calculate()

        if self._memory:
            self.save(self._count)
            self._count += 1

        return self._latest_value
