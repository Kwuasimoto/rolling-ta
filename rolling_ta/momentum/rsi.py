import pandas as pd
import numpy as np

from rolling_ta.indicator import Indicator


# Math derived from chatGPT + https://www.investopedia.com/terms/r/rsi.asp
class RSI(Indicator):
    """Rolling RSI Indicator https://www.investopedia.com/terms/r/rsi.asp"""

    _gains = 0.0
    _losses = 0.0
    _prev_price = np.nan
    _avg_gain = np.nan
    _avg_loss = np.nan
    _rsi = np.nan

    def __init__(self, data: pd.DataFrame, period: int = 14, memory=True) -> None:
        """Rolling RSI Indicator
        https://www.investopedia.com/terms/r/rsi.asp

        Args:
            series (pd.DataFrame): The initial dataframe or list of information to use. Must contain a "close" column.
            period (int): Default=14 | RSI Window.
            memory (bool): Default=True | Memory flag, if false removes all information not required for rsi.update().
        """
        super().__init__(data, period)

        close = data["close"]
        delta = close.diff()

        # Use numpy vectorization for initial gains and losses.
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # SMA-like phase-1 calculations
        self._avg_gain = np.mean(gains[:period])
        self._avg_loss = np.mean(losses[:period])

        # Store the rsi
        self._rsi = self.calculate()

        # Initialize state for rolling updates.
        self._gains = np.sum(gains)
        self._losses = np.sum(losses)
        self._prev_price = close.iloc[-1]
        self._count = len(data)

        # Remove information not necessary for updates if not flag.
        if not memory:
            self._data = None
        else:
            self._data["rsi"] = np.nan
            self._data.at[period - 1, "rsi"] = self._rsi

        if len(data) >= period:
            for i in range(period, self._count):
                gain = gains[i]
                loss = losses[i]

                self._avg_gain = (self._avg_gain * (period - 1) + gain) / period
                self._avg_loss = (self._avg_loss * (period - 1) + loss) / period

                self._rsi = self.calculate()

                if memory:
                    self._data.at[i, "rsi"] = self._rsi

    def data(self):
        return self._data["rsi"]

    def calculate(self):
        if self._avg_loss == 0:
            return 100  # Avoid division by 0

        rs = self._avg_gain / self._avg_loss
        return 100 - (100 / (1 + rs))

    def update(self, close: float):
        # Get the delta in price, and calculate gain/loss
        delta = close - self._prev_price
        self._prev_price = close

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
        self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period
        self._rsi = self.calculate()
