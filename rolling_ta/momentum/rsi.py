import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator


# Math derived from chatGPT + https://www.investopedia.com/terms/r/rsi.asp
class RSI(Indicator):
    """Rolling RSI Indicator https://www.investopedia.com/terms/r/rsi.asp"""

    _gains = 0.0
    _losses = 0.0
    _prev_price = np.nan
    _avg_gain = np.nan
    _avg_loss = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int = 14,
        memory: bool = True,
        init: bool = True,
        roll: bool = True,
    ) -> None:
        """Rolling RSI indicator

        https://www.investopedia.com/terms/r/rsi.asp

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=14 | Window length.
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
            roll (bool, optional): Default=True | Calculate remaining indicator values upon instantiation.
        """
        super().__init__(data, period, memory, init, roll)

        self.set_column_names({"rsi": f"rsi_{self._period}"})

        if init:
            self.init()

    def init(self):
        close = self._data["close"]
        count = close.shape[0]
        column = self._column_names["rsi"]

        delta = close.diff()

        # Use numpy vectorization for initial gains and losses.
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # SMA-like phase-1 calculations
        self._avg_gain = np.mean(gains[: self._period])
        self._avg_loss = np.mean(losses[: self._period])

        # Store the rsi
        self._latest_value = self.calculate()

        # Initialize state for rolling updates.
        self._gains = np.sum(gains)
        self._losses = np.sum(losses)

        # Remove information not necessary for updates if not flag.
        if self._memory:
            self._count = count
            self._data[column] = np.nan
            self._data.at[self._period - 1, column] = self._latest_value
        else:
            self._data = None

        # Roll rest of RSI, don't use update to avoid function overhead.
        if self._roll:
            for i in range(self._period, count):
                gain = gains[i]
                loss = losses[i]

                self._avg_gain = (
                    self._avg_gain * (self._period - 1) + gain
                ) / self._period
                self._avg_loss = (
                    self._avg_loss * (self._period - 1) + loss
                ) / self._period

                self._latest_value = self.calculate()

                if self._memory:
                    self._data.at[i, column] = self._latest_value

    def update(self, close: float):
        # Get the delta in price, and calculate gain/loss
        delta = close - self._prev_price
        self._prev_price = close

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
        self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period
        self._latest_value = self.calculate()

        if self._memory:
            self._data.at[self._count, self._column_names["rsi"]] = self._latest_value
            self._count += 1

    def calculate(self):
        if self._avg_loss == 0:
            return 100  # Avoid division by 0

        rs = self._avg_gain / self._avg_loss
        return 100 - (100 / (1 + rs))
