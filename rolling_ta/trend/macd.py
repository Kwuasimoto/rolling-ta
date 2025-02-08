import pandas as pd
from rolling_ta.indicator import Indicator
from typing import Dict, Optional


class MACD(Indicator):

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        period_config: Dict[str, int] = {"fast": 12, "slow": 26, "smoothing": 9},
        memory: bool = True,
        retention: Optional[int] = None,
        columns: Optional[list[str]] = None,
        init: bool = False,
        force: bool = False,
        initialization_state: bool = False,
    ) -> None:
        super().__init__(data, period_config, memory, retention, columns, init)
        if self._init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: Optional[bool] = True):
        super().calc(
            force=force,
            initialization_state=initialization_state,
        )

    def update(self, data: pd.Series):
        return super().update(data)
