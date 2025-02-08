from array import array
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from rolling_ta.extras.numba import _rsi, _rsi_update
from rolling_ta.indicator import Indicator

RelativeStrengthIndexKeys = Literal["rsi"]
RelativeStrengthIndexPeriods = Literal["rsi"]


class RSI(Indicator):
    """
    Relative Strength Index (RSI) indicator.

    The RSI is a momentum oscillator that measures the speed and change of price
    movements. It oscillates between 0 and 100 and is used to identify overbought
    or oversold conditions in an asset. This class calculates the RSI using
    historical price data over a specified period.

    Material
    --------
        https://www.investopedia.com/terms/r/rsi.asp
    """

    _keys: List[RelativeStrengthIndexKeys] = ["rsi"]
    _period_config: Dict[RelativeStrengthIndexPeriods, int] = {"rsi": 14}

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        keys: List[RelativeStrengthIndexKeys] = _keys,
        period_config: Dict[RelativeStrengthIndexPeriods, int] = _period_config,
        memory: bool = True,
        retention: Optional[int] = None,
        init: bool = False,
        force: bool = False,
        initialization_state: bool = False,
    ) -> None:
        """
        Initialize the RSI indicator.

        Args:
            data (pd.DataFrame): The initial dataframe containing price data with a 'close' column.
            period (int): Default=14 | The period over which to calculate the RSI.
            memory (bool): Default=True | Whether to store RSI values in memory.
            retention (int): Default=20000 | The maximum number of RSI values to store in memory
            init (bool): Default=True | Whether to calculate the initial RSI values upon instantiation.
        """
        super().__init__(
            data,
            keys=keys,
            period_config=period_config,
            memory=memory,
            retention=retention,
            init=init,
            force=force,
            initialization_state=initialization_state,
        )
        self.alpha = 1 / period_config["rsi"]
        if init:
            self.calc(
                force=force,
                initialization_state=initialization_state,
            )

    def calc(self, force: bool = False, initialization_state: bool = True):
        if self._initialized and not force:
            return

        close = self._data["close"].to_numpy(np.float64)
        rsi = np.zeros(close.size, dtype=np.float64)
        gains = np.zeros(close.size, dtype=np.float64)
        losses = np.zeros(close.size, dtype=np.float64)

        rsi, avg_gain, avg_loss, close_p = _rsi(
            close=close,
            rsi_container=rsi,
            gains_container=gains,
            losses_container=losses,
            period=self._period_config["rsi"],
            p_1=self._period_config["rsi"] - 1,
        )

        if self._memory:
            self._rsi = array("f", rsi)

        self._avg_gain = avg_gain
        self._avg_loss = avg_loss
        self._close_p = close_p

        self.drop_data()
        self._set_initialized(state=initialization_state)

        return self

    def update(self, data: pd.Series):
        close = data["close"]

        rsi, avg_gain, avg_loss = _rsi_update(
            close,
            self._close_p,
            self._avg_gain,
            self._avg_loss,
            self.alpha,
        )

        self._avg_gain = avg_gain
        self._avg_loss = avg_loss
        self._close_p = close

        if self._memory:
            self._rsi.append(rsi)

        return self

    def fit(
        self,
        data,
        period_config: Dict[RelativeStrengthIndexPeriods, int] = _period_config,
    ):
        super().fit(data, period_config)

    def to_array(self, get: RelativeStrengthIndexKeys = "rsi"):
        return super().to_array(get)

    def to_numpy(
        self,
        get: RelativeStrengthIndexKeys = "rsi",
        dtype: Optional[np.dtype] = np.float64,
        **kwargs,
    ):
        return super().to_numpy(get, dtype, **kwargs)

    def to_series(
        self,
        get: RelativeStrengthIndexKeys = "rsi",
        dtype: Optional[type] = float,
        name: Optional[str] = None,
        **kwargs,
    ):
        return super().to_series(get, dtype, name, **kwargs)
