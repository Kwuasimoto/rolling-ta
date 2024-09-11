import pandas as pd
from rolling_ta.indicator import Indicator


class MACD(Indicator):

    def __init__(
        self, data: pd.DataFrame, period: int, memory: bool, init: bool
    ) -> None:
        super().__init__(data, period, memory, init)

        if self._init:
            self.init()

    def init(self):
        pass

    def update(self, data: pd.Series):
        return super().update(data)
