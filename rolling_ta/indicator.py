import pandas as pd


class Indicator:

    _data: pd.DataFrame
    _period: int
    _memory: bool
    _init: bool
    _count = 0

    def __init__(
        self, data: pd.DataFrame, period: int, memory: bool = True, init: bool = False
    ) -> None:
        if len(data) < period:
            raise ArithmeticError(
                "len(data) must be greater than, or equal to the period."
            )

        self._data = data
        self._period = period
        self._memory = memory
        self._init = init

    def init(self):
        pass

    def calculate(self):
        pass

    def update(self):
        pass

    def data(self):
        return self._data
