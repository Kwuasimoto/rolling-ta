from array import array
from enum import Enum
import numpy as np
import pandas as pd
from typing import Any, List, Literal, Optional, Union, Dict

from rolling_ta.logging import log


class IndicatorID(Enum):
    base = 0
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

    _id: IndicatorID = IndicatorID.base
    _data: pd.DataFrame = None
    _keys: List[Literal["base"]] = ["base"]
    _period_config: Dict[Literal["base"], int] = {}
    _memory: bool = True
    _retention: Optional[int] = None
    _init: bool = False
    _initialized: bool = False
    _count = 0

    def __init__(
        self,
        data: Optional[pd.DataFrame],
        keys: List[Literal["base"]],
        period_config: Dict[Literal["base"], int],
        memory: bool,
        retention: Optional[int],
        init: bool,
        force: bool,
        initialization_state: bool,
    ) -> None:
        if data is not None:
            self._data = data.copy(deep=True)

        self._keys = keys
        self._period_config = period_config
        self._memory = memory
        self._retention = retention
        self._init = init

    def get_config(self, key: str = None):
        cfg = {
            "period_config": self._period_config,
            "memory": self._memory,
            "retention": self._retention,
            "init": self._init,
            "count": self._count,
        }
        if key is None:
            return cfg
        return cfg[key]

    def set_period_config(
        self, period_config: Optional[Dict[Literal["base"], Any]] = _period_config
    ):
        for k, v in period_config.items():
            if k in self._period_config:
                self._period_config[k] = v

    def extend_period_config(self, ext: Dict[Literal["base_ext"], Any]):
        self._period_config.update()

    def set_keys(self, keys: list[str]):
        self._keys = keys

    def validate_data(self, data: pd.DataFrame):
        """Checks if the input data is compatible with the indicator configuration."""
        if self._period_config is None:
            log.error(f"Unable to determine if data is valid if period_config is None")
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

    def fit(
        self,
        data: pd.DataFrame,
        period_config: Optional[Dict[Literal["base"], int]] = _period_config,
    ):
        if period_config is not None:
            log.debug(f"Fitting period: {period_config}", self)
            self.set_period_config(period_config)
        # Validate period input
        if self.validate_data(data):
            log.debug(
                f"Validated data for {self.__class__.__name__}, fitting shape={data.shape}",
                self,
            )
            self._data = data.copy(deep=True)
        else:
            raise ValueError(f"A dataframe incompatible with {self} was supplied!")

    def period(self, key: Literal["base"] = "base"):
        if key not in self._period_config:
            raise ValueError(
                "Invalid key for Indicator period_config! Please review the indicator subclass period configuration for details. \nThe python help(indicator) function will display the class doc_string with the required period config dictionary."
            )
        if isinstance(self._period_config, dict):
            return self._period_config[key]
        return self._period_config

    def calc(
        self, force: bool = False, initialization_state: bool = False
    ) -> "Indicator":
        """Behavior:

        - Returns early if self._initialized is True
        - Controls initialization state (recalculation guard) with 'initialization_state' parameter.
        - Can be forced to recalculate with 'force' parameter.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} Indicator not implemented yet! sorry!"
        )

    def update(self, data: pd.Series) -> "Indicator":
        raise NotImplementedError(
            f"{self.__class__.__name__} Indicator update function not implemented yet! sorry!"
        )

    def get(self, index: int, key: Optional[Literal["unknown"]] = None):
        return getattr(self, f"_{key}")[index]

    def apply_retention(self): ...

    def _set_initialized(self, state=True):
        """Shallow version of set_initialized"""
        self._initialized = state

    def set_initialized(self, state=True):
        """Propagates to nested indicators!"""
        self._set_initialized(state)
        for key in self._keys:
            fkey = f"_{key}"
            if hasattr(self, fkey):
                attr = getattr(self, fkey)
                if isinstance(attr, Indicator):
                    attr.set_initialized(state)

    def initialized(self):
        return self._initialized

    def drop_data(self):
        """Drops the data used to calculate the indicator."""
        self._data = None

    def drop_values(self):
        """Drops the calculated values in memory."""
        for key in self._keys:
            fkey = f"_{key}"
            if hasattr(self, fkey):
                attr = getattr(self, fkey)
                if isinstance(attr, Indicator):
                    attr.drop_values()
                if isinstance(attr, array):
                    delattr(self, fkey)

    def to_array(self, get: Literal["base"] = "base"):
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
        get: Literal["base"] = "base",
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
        get: Literal["base"] = "base",
        dtype: Union[type, None] = float,
        name: Optional[str] = None,
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
        return pd.Series(
            raw,
            dtype=dtype,
            name=f"{get}_{self._period_config[get]}" if name is None else name,
            **kwargs,
        )

    def to_dataframe(self, dtype: Optional[type] = float):
        data_objs = {}
        for key in self._keys:
            if key in self._period_config:
                period = self._period_config[key]
                data_objs[f"{self.__class__.__name__}_{key}_{period}"] = self.to_numpy(
                    get=key, dtype=dtype
                )
        return pd.DataFrame(data_objs)


#  and isinstance(getattr(self, f"_{k}"), Iterable)
