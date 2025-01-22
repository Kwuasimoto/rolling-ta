from enum import Enum
import numpy as np
import pandas as pd
from typing import Literal, Optional, Union, Dict

from rolling_ta.logging import log


class IndicatorID(Enum):
    # Momentum (1xxx - 1999)
    bop = 1000
    rsi = 1001
    stoch_rsi = 1002

    # Trend (2xxx - 2999)
    adx = 2000
    dmi = 2001
    donchian = 2002
    ema = 2003
    hma = 2004
    lr = 2005
    lrf = 2006
    lrr = 2007
    macd = 2008
    sma = 2009
    wma = 2010

    # Volatility (3xxx - 3999)
    atr = 3000
    bb = 3001
    tr = 3002

    # Volume (4xxx - 4999)
    mfi = 4000
    obv = 4001
    vwap = 4002

    # Omni (5xxx - 5999)
    ichimoku_cloud = 5000


class Indicator:

    _id: IndicatorID
    _data: pd.DataFrame
    _period_config: Union[int, Dict[str, int]] = None
    _period_default: Union[int, Dict[str, int]] = None  # Set by the subclass
    _memory: bool
    _retention: Optional[int]
    _init: bool
    _initialized: bool = False
    _count = 0

    def get_config(self, key: str = None):
        cfg = {
            "id": self._id,
            "period_config": self._period_config,
            "memory": self._memory,
            "retention": self._retention,
            "init": self._init,
            "count": self._count,
        }
        if key is None:
            return cfg
        return cfg[key]

    def __init__(
        self,
        data: Optional[pd.DataFrame],
        period_config: Union[int, Dict[str, int]],
        memory: bool,
        retention: Union[int, None],
        init: bool,
    ) -> None:
        self._data = data
        self._period_config = period_config
        self._memory = memory
        self._retention = retention
        self._init = init

        # Check if _period_default set
        if self._period_default is not None:
            if isinstance(self._period_config, Dict):
                for k, v in self._period_default.items():
                    if k not in self._period_config:
                        self._period_config[k] = v

    def validate_data(self, data: pd.DataFrame):
        """Checks if the input data is compatible with the indicator configuration."""
        if isinstance(self._period_config, int):
            if len(data) < self._period_config:
                raise ValueError(
                    f"len(data) must be greater than, or equal to the period. [len(data)={len(data)}, period={self._period_config}]"
                )
        elif isinstance(self._period_config, dict):
            for [key, period] in self._period_config.items():
                if len(data) < period:
                    raise ValueError(
                        f"len(data) must be greater than, or equal to each period. \n[Key={key}, Period={period}, Data_Len={len(data)}]"
                    )

        return True

    def fit(self, data: pd.DataFrame):
        log.debug(f"Trying to fit : {data.describe()}")
        # Validate period input
        if self.validate_data(data):
            log.debug(f"Fitting {len(data)} data points.")
            self._data = data
        else:
            raise ValueError(f"A dataframe incompatible with {self} was supplied!")

    def period(self, key: Union[str, None] = None):
        if key is not None and key not in self._period_config:
            raise ValueError(
                "Invalid key for Indicator period_config! Please review the indicator subclass period configuration for details. \nThe python help(indicator) function will display the class doc_string with the required period config dictionary."
            )
        if isinstance(self._period_config, dict):
            return self._period_config[key]
        return self._period_config

    def init(self, __name__: str = "Unknown"):
        raise NotImplementedError("Indicator not implemented yet! sorry!")

    def update(self, data: pd.Series, __name__: str = "Unknown"):
        raise NotImplementedError(
            "Indicator update function not implemented yet! sorry!"
        )

    def apply_retention(self): ...

    def set_initialized(self, state=True):
        self._initialized = state

    def initialized(self):
        return self._initialized

    def drop_data(self):
        self._data = None

    def to_array(self, get: Literal["unknown"] = "unknown"):
        """Returns the raw information associated with this indicator object.

        Args:
            get (literal, optional): Default indicator | the indicator data associated with the get key.

        Returns:
            array: data array
        """
        raw = getattr(self, f"_{get}", None)
        assert (
            raw is not None
        ), f"Indicator does not exist, memory may not set, or get value is incorrect. [get={get}, memory={self._memory}]"
        return raw

    def to_numpy(
        self,
        get: Literal["Unknown"] = "unknown",
        dtype: Union[np.dtype, None] = np.float64,
        **kwargs,
    ):
        """Returns the information associated with this indicator object as a numpy array.

        Args:
            get (literal, optional): Default indicator | the indicator data associated with the get key.
            dtype (dtype, optional): Default np.float64 | dtype for numpy array.
            kwargs (dict, optional): Default None | extra arguments for np.array()

        Returns:
            ndarray: data ndarray.
        """
        raw = getattr(self, f"_{get}", None)
        assert (
            raw is not None
        ), f"Indicator does not exist, memory may not set, or get value is incorrect. [get={get}, memory={self._memory}]"
        return np.array(raw, dtype=dtype, **kwargs)

    def to_series(
        self,
        get: Literal["Unknown"] = "unknown",
        dtype: Union[type, None] = float,
        name: Union[str, None] = None,
        **kwargs,
    ):
        """Returns the information associated with this indicator object as a pandas series.

        Args:
            get (literal, optional): Default indicator | the indicator data associated with the get key.
            dtype (dtype, optional): Default float | dtype for pandas array.
            name (str, optional): Default None | name of series.
            kwargs (dict, optional): Default None | extra arguments for pd.Series()

        Returns:
            series: LR2 series.
        """
        raw = getattr(self, f"_{get}", None)
        assert (
            raw is not None
        ), f"Indicator does not exist, memory may not set, or get value is incorrect. [get={get}, memory={self._memory}]"
        return pd.Series(raw, dtype=dtype, name=name, **kwargs)
