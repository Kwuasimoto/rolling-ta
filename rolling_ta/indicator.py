import numpy as np
import pandas as pd


class Indicator:

    _data: pd.DataFrame
    _period: int
    _memory: bool
    _init: bool
    _roll: bool
    _column_names: dict
    _count = 0
    _latest_value = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period: int,
        memory: bool,
        init: bool,
        roll: bool,
    ) -> None:
        if len(data) < period:
            raise ArithmeticError(
                "len(data) must be greater than, or equal to the period."
            )

        self._data = data
        self._period = period
        self._memory = memory
        self._init = init
        self._roll = roll

    def init(self):
        pass

    def update(self):
        pass

    def calculate(self):
        pass

    def data(self):
        if not self._memory:
            raise MemoryError(
                "Memory is set to false so Dataframe was deleted. Use .latest_value() to get the most recent value or set memory to True"
            )
        return self._data

    def latest_value(self):
        return self._latest_value

    def set_column_names(self, names: dict):
        self._column_names = names
