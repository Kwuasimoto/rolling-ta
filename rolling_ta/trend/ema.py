from array import array
import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _ema, _ema_update
from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger


class NumbaEMA(Indicator):

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
        ema = _ema(close, self._weight, self._period_config)

        self._ema_latest = ema[-1]

        if self._memory:
            self._ema = array("f", ema)

        self.drop_data()

    def update(self, data: pd.Series):
        self._ema_latest = _ema_update(data["close"], self._weight, self._ema_latest)

        if self._memory:
            self._ema = np.append(self._ema, self._ema_latest)

    def ema(self):
        if not self._memory:
            raise MemoryError("NumbaEMA._memory = False")
        return pd.Series(self._ema)


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) Indicator.

    The EMA gives more weight to recent prices, making it more responsive to new information compared to the Simple Moving Average (SMA).
    This indicator is commonly used to identify trends and smooth out price data.

    Material
    --------
        https://www.investopedia.com/terms/e/ema.asp

    Attributes
    ----------
    _close : float
        The most recent closing price used for calculating the EMA.
    _ema : pd.Series
        A pandas Series storing the calculated EMA values for each period.
    _ema_latest : float
        The latest EMA value.
    _weight : float
        The weight for the EMA calculation, derived from the period.

    Methods
    -------
    **__init__(data: pd.DataFrame, period: int = 14, memory: bool = True, init: bool = True)** -> None

        Initializes the EMA indicator with the given data, period, and options.

    **init()** -> None

        Calculates the initial EMA values based on the provided data.

    **update(data: pd.Series)** -> None

        Updates the EMA based on new incoming data.

    **ema()** -> pd.Series

        Returns the stored EMA values if memory is enabled.
        Throws a MemoryError if memory=False

    **ema_latest()** -> float

        Returns the latest EMA value.
    """

    _close = np.nan

    _ema: pd.Series
    _ema_latest = np.nan

    _weight = np.nan

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: int = 14,
        weight: np.float64 = 2.0,
        memory: bool = True,
        retention: int = 20000,
        init: bool = True,
    ) -> None:
        """
        Initializes the EMA indicator with the given data, period, weight, and options.

        Args:
            data (pd.DataFrame): The initial dataframe to use. Must contain a "close" column.
            period (int): Default=14 | Window length for the EMA calculation.
            weight (np.float64, optional): Default=2.0 | The weight of the EMA's multiplier.
            retention (int): Default=20000 | The maximum number of RSI values to store in memory
            memory (bool): Default=True | Memory flag, if false removes all information not required for updates.
            init (bool, optional): Default=True | Calculate the immediate indicator values upon instantiation.
        """
        super().__init__(data, period_config, memory, retention, init)
        logger.debug(
            f"EMA init: [data_len={len(data)}, period={period_config}, memory={memory}]"
        )

        # EX: (2 / (10 + 1))
        self._weight = weight / (period_config + 1)

        if init:
            self.init()

    def init(self):
        """
        Calculates the initial EMA values based on the provided data.

        This method computes the EMA using the provided data and initializes internal attributes. If memory is enabled, it also stores the computed EMA values.
        """
        close = self._data["close"]
        ema = np.zeros_like(close, dtype=np.float64)

        self._ema_latest = np.sum(close[: self._period_config]) / self._period_config

        ema[self._period_config - 1] = self._ema_latest
        ema[self._period_config :] = close[self._period_config :].apply(self._calc)

        # Use Memory
        if self._memory:
            self._count = close.shape[0]
            self._ema = pd.Series(ema)

            # if self._count > self._retention:
            #     self.apply_retention()

        self.drop_data()

    def _calc(self, close: np.float64):
        self._ema_latest = (
            (close - self._ema_latest) * self._weight
        ) + self._ema_latest
        return self._ema_latest

    def update(self, data: pd.Series):
        """
        Updates the EMA based on new incoming closing price data.

        Args:
            close (np.float64): The new closing price to update the EMA with.
        """
        close = data["close"]
        self._ema_latest = (
            (close - self._ema_latest) * self._weight
        ) + self._ema_latest

        if self._memory:
            self._ema[self._count] = self._ema_latest
            self._count += 1

    def apply_retention(self):
        self._ema = self.apply_retention()

    def ema(self):
        """
        Returns the stored EMA values.

        Returns:
            pd.Series: A pandas Series containing the EMA values.

        Raises:
            MemoryError: If function called and memory=False
        """
        if not self._memory:
            raise MemoryError("EMA._memory = False")
        return self._ema

    def ema_latest(self):
        """
        Returns the latest EMA value.

        Returns:
            float: The most recent EMA value.
        """
        return self._ema_latest
