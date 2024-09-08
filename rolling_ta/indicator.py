import pandas as pd


class Indicator:

    _data: pd.DataFrame
    _period: int
    _count = 0

    def __init__(self, data: pd.DataFrame, period: int) -> None:
        if len(data) < period:
            raise ArithmeticError(
                "len(data) must be greater than, or equal to the period."
            )

        self._data = data
        self._period = period

    def calculate(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def data(self):
        return self._data
