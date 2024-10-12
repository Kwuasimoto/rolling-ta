import pandas as pd
from rolling_ta.indicator import Indicator
from typing import Dict


class MACD(Indicator):

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Dict[str, int] = {"fast": 12, "slow": 26, "smoothing": 9},
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if self._init:
            self.init()

    def init(self):
        super().init(__name__)

    def update(self, data: pd.Series):
        return super().update(data, __name__)
