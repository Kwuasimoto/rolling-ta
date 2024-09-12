from typing import Dict, Union
import numpy as np
import pandas as pd
from rolling_ta.indicator import Indicator
from collections import deque


class IchimokuCloud(Indicator):
    """
    Ichimoku Cloud indicator, a comprehensive tool used in technical analysis
    to assess trends, support/resistance levels, and momentum.

    The Ichimoku Cloud calculates five key components:
    1. Tenkan-sen (Conversion Line)
    2. Kijun-sen (Base Line)
    3. Senkou Span A (Leading Span A)
    4. Senkou Span B (Leading Span B)
    5. Chikou Span (Lagging Span).

    Required Dictionary Keys:
    -------------------------
    The `periods` dictionary passed to the class must contain the following keys:

    - 'tenkan': (int) Period for the Tenkan-sen (conversion line), typically 9.
    - 'kijun': (int) Period for the Kijun-sen (base line), typically 26.
    - 'senkou': (int) Period for the Senkou Span B (leading span B), typically 52.

    Optional Dictionary Key:
    ------------------------
    - 'lagging': (int, optional) Custom period for the Chikou Span (lagging span).
      If not provided, it defaults to the 'kijun' period (typically 26).

    Attributes:
    -----------

    Example:
    --------
    >>> period_config = {'tenkan': 9, 'kijun': 26, 'senkou_span': 52, 'lagging': 30}
    >>> ichimoku = IchimokuCloud(data, period_config)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Dict[str, int] = {"tenkan": 9, "kijun": 26, "senkou": 52},
        memory: bool = True,
        retention: Union[int, None] = 20000,
        init: bool = True,
    ) -> None:
        super().__init__(data, period_config, memory, retention, init)

        if not isinstance(self._period_config, dict):
            raise ValueError(
                "Ichimoku Cloud period config must be a dictionary. \nPlease review the docstring or use help(indicator) for more information."
            )

        # Deconstruct period config to attributes to avoid function overhead
        self._tenkan_period = self.period("tenkan")
        self._kijun_period = self.period("kijun")
        self._senkou_period = self.period("senkou")
        self._lagging_period = (
            self.period("lagging")
            if "lagging" in self._period_config
            else self._kijun_period
        )

        if self._init:
            self.init()

    def init(self):
        high = self._data["high"]
        low = self._data["low"]
        close = self._data["close"]

        # 1. Calculate tenkan and kijun
        tenkan_highs = high.rolling(
            self._tenkan_period, min_periods=self._tenkan_period
        ).max()
        tenkan_lows = low.rolling(
            self._tenkan_period, min_periods=self._tenkan_period
        ).min()
        tenkan = (tenkan_highs + tenkan_lows) * 0.5

        kijun_highs = high.rolling(
            self._kijun_period, min_periods=self._tenkan_period
        ).max()
        kijun_lows = low.rolling(
            self._kijun_period, min_periods=self._tenkan_period
        ).min()
        kijun = (kijun_highs + kijun_lows) * 0.5

        senkou_a = (tenkan + kijun) * 0.5

        senkou_b_highs = high.rolling(
            self._senkou_period, min_periods=self._senkou_period
        ).max()
        senkou_b_lows = low.rolling(
            self._senkou_period, min_periods=self._senkou_period
        ).min()
        senkou_b = (senkou_b_highs + senkou_b_lows) * 0.5

        lagging = close.shift(-self._lagging_period)

        period_max = max(
            [
                self._tenkan_period,
                self._kijun_period,
                self._senkou_period,
                self._lagging_period,
            ]
        )

        if self._memory:
            self._count = period_max
            self._tenkan = tenkan.tail(self._retention)
            self._kijun = kijun.tail(self._retention)
            self._senkou_a = senkou_a.tail(self._retention)
            self._senkou_b = senkou_b.tail(self._retention)
            self._lagging = lagging.tail(self._retention)

        self._tenkan_latest = tenkan.iat[-1]
        self._kijun_latest = kijun.iat[-1]
        self._senkou_a_latest = senkou_a.iat[-1]
        self._senkou_b_latest = senkou_b.iat[-1]
        self._lagging_latest = lagging.iat[-self._lagging_period]

        # Store related information for required updates
        self._high = deque(high, maxlen=period_max)
        self._low = deque(low, maxlen=period_max)
        self._close = deque(close, maxlen=period_max)

        self.drop_data()

    def update(self, data: pd.Series):
        self._high.append(data["high"])
        self._low.append(data["low"])
        self._close.append(data["close"])

        high = pd.Series(self._high)
        low = pd.Series(self._low)

        tenkan_period_index = -self._tenkan_period
        kijun_period_index = -self._kijun_period
        senkou_period_index = -self._senkou_period

        tenkan_high = np.max(high.iloc[tenkan_period_index:])
        tenkan_low = np.min(low.iloc[tenkan_period_index:])
        tenkan = (tenkan_high + tenkan_low) * 0.5

        kijun_high = np.max(high.iloc[kijun_period_index:])
        kijun_low = np.min(low.iloc[kijun_period_index:])
        kijun = (kijun_high + kijun_low) * 0.5

        senkou_a = (tenkan + kijun) * 0.5

        senkou_high = np.max(high.iloc[senkou_period_index:])
        senkou_low = np.min(low.iloc[senkou_period_index:])
        senkou_b = (senkou_high + senkou_low) * 0.5

        self._lagging_latest = self._close[-self._lagging_period]

        if self._memory:
            self._tenkan[self._count] = tenkan
            self._kijun[self._count] = kijun
            self._senkou_a[self._count] = senkou_a
            self._senkou_b[self._count] = senkou_b
            self._lagging[self._count] = self._lagging_latest

            self._count += 1

            if self._retention:
                self._tenkan = self.apply_retention(self._tenkan)
                self._kijun = self.apply_retention(self._tenkan)
                self._senkou_a = self.apply_retention(self._senkou_a)
                self._senkou_b = self.apply_retention(self._senkou_b)
                self._lagging = self.apply_retention(self._lagging)

    def tenkan(self):
        return self._tenkan

    def tenkan_latest(self):
        return self._tenkan_latest

    def kijun(self):
        return self._kijun

    def kijun_latest(self):
        return self._kijun_latest

    def senkou_a(self):
        return self._senkou_a

    def senkou_a_latest(self):
        return self._senkou_a_latest

    def senkou_b(self):
        return self._senkou_b

    def senkou_b_latest(self):
        return self._senkou_b_latest

    def lagging(self):
        return self._lagging

    def lagging_latest(self):
        return self._lagging_latest
