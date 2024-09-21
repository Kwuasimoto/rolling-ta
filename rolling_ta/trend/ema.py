from array import array
import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _ema, _ema_update
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) Indicator.

    The EMA gives more weight to recent prices, making it more responsive to new information compared to the Simple Moving Average (SMA).
    This indicator is commonly used to identify trends and smooth out price data.

    Material
    --------
        https://www.investopedia.com/terms/e/ema.asp
    """

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = 14,
        weight: np.float64 = 2.0,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)
        self._weight = weight / (period_config + 1)
        if self._init:
            self.init()

    def init(self):
        close = self._data["close"].to_numpy(dtype=np.float64)
        ema = np.zeros(close.size)

        ema, ema_latest = _ema(
            close,
            ema,
            self._weight,
            self._period_config,
        )

        self._ema_latest = ema_latest

        if self._memory:
            self._ema = array("f", ema)

        self.drop_data()

    def update(self, data: pd.Series):
        self._ema_latest = _ema_update(data["close"], self._weight, self._ema_latest)

        if self._memory:
            self._ema.append(self._ema_latest)

    def ema(self):
        if not self._memory:
            raise MemoryError("NumbaEMA._memory = False")
        return pd.Series(self._ema)
