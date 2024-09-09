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

        self.set_column_names(
            {
                "sma": f"sma_{period}",
                "lower_band": f"bb_lband_{self._period}",
                "upper_band": f"bb_uband_{self._period}",
            }
        )
        self._weight = weight

        if init:
            self.init()

    def init(self):
        # We want the SMA to roll with the BollingerBands, not independently.
        close = self._data["close"]
        count = close.shape[0]

        sma_column = self._column_names["sma"]
        lband_column = self._column_names["lower_band"]
        uband_column = self._column_names["upper_band"]

        self._sma = SMA(self._data, self._period, False, self._init, False)

        # Calculate initital BB Values
        self.calculate()

        if self._memory:
            self._count = count
            self._data[sma_column] = np.nan
            self._data[lband_column] = np.nan
            self._data[uband_column] = np.nan
            self._data.at[self._period - 1, sma_column] = self._sma._latest_value
            self._data.at[self._period - 1, uband_column] = self._upper_band
            self._data.at[self._period - 1, lband_column] = self._lower_band
        else:
            self._data = None
            self._sma._data = None

        # Roll the rest of the BB
        if self._roll:
            for i in range(self._period, count):
                current_close = close[i]
                self._sma.update(current_close)

                self._stddev = np.std(self._sma._window)
                stddev_weighted = self._stddev * self._weight

                self._upper_band = self._sma._latest_value + stddev_weighted
                self._lower_band = self._sma._latest_value - stddev_weighted

                if self._memory:
                    self._data.at[i, sma_column] = self._sma._latest_value
                    self._data.at[i, uband_column] = self._upper_band
                    self._data.at[i, lband_column] = self._lower_band

    def update(self, close: np.float64):
        # Update SMA
        self._sma.update(close)

        # Calculate initital BB Values
        self.calculate()

        if self._memory:
            self._data.at[self._count, self._column_names["sma"]] = (
                self._sma._latest_value
            )
            self._data.at[self._count, self._column_names["upper_band"]] = (
                self._upper_band
            )
            self._data.at[self._count, self._column_names["lower_band"]] = (
                self._lower_band
            )
            self._count += 1

        return self._latest_value

    def calculate(self):
        self._stddev = np.std(self._sma._window)
        stddev_weighted = self._stddev * self._weight

        self._upper_band = self._sma._latest_value + stddev_weighted
        self._lower_band = self._sma._latest_value - stddev_weighted
