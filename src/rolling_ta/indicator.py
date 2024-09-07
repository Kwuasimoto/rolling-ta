import pandas as pd


class Indicator:

    _data: pd.DataFrame
    _period: int

    def __init__(self, data: pd.DataFrame, period: int) -> None:
        self._data = data
        self._period = period

    def calculate(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass
