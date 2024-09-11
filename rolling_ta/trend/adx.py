from pandas import DataFrame, Series
from rolling_ta.indicator import Indicator


class ADX(Indicator):

    def __init__(self, data: DataFrame, period: int, memory: bool, init: bool) -> None:
        super().__init__(data, period, memory, init)

        if self._init:
            self.init()

    def init(self):
        return super().init()

    def update(self, data: Series):
        return super().update(data)
