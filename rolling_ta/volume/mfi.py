from pandas import DataFrame
from rolling_ta.indicator import Indicator


class MFI(Indicator):

    def __init__(
        self,
        data: DataFrame,
        period: int,
        memory: bool,
        init: bool,
        roll: bool,
    ) -> None:
        super().__init__(data, period, memory, init, roll)

        if self._init:
            self.init()

    def init(self):

        if self._roll:
            pass

    def update(self):
        return super().update()

    def calculate(self):
        return super().calculate()
