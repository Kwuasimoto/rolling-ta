import pandas as pd

from typing import List

from rolling_ta.indicator import Indicator


# Math derived from chatGPT + https://www.investopedia.com/terms/r/rsi.asp
class RSI(Indicator):

    _gains = 0.0
    _losses = 0.0
    _prev_price = None
    _avg_gain = None
    _avg_loss = None
    _rsi = None
    _count = 0

    def __init__(self, data: pd.DataFrame, period: int) -> None:
        """Rolling RSI Indicator

        Args:
            series (pd.DataFrame): The initial dataframe or list of information to use.
            period (int): RSI Window
        """
        super().__init__(data, period)

    def calculate(self):
        pass

    def update(self, price: float):
        # We can't calculate RSI with a single data point, so return None indicating no calculation happened.
        if self._prev_price is None:
            self._prev_price = price
            return None

        # Get the change in price, and calculate gain/loss
        change = price - self._prev_price
        gain = max(0, change)
        loss = max(0, -change)
