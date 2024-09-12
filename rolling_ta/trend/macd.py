import pandas as pd
from rolling_ta.indicator import Indicator


class MACD(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = {"fast": 12, "slow": 26, "smoothing": 9},
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        pass

    def update(self, data: pd.Series):
        return super().update(data)
