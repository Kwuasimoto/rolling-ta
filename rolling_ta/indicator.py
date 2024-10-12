import numpy as np
import pandas as pd
from typing import Literal, Union, Dict


class Indicator:

    _data: pd.DataFrame
    _period_config: Union[int, Dict[str, int]]
    _period_default: Union[int, Dict[str, int]] = None  # Set by the subclass
    _memory: bool
    _retention: Union[int, None]
    _init: bool
    _initialized: bool = False
    _count = 0

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Union[int, Dict[str, int]],
        memory: bool,
        retention: Union[int, None],
        init: bool,
    ) -> None:

        # Check if _period_default set
        if self._period_default is not None:
            if isinstance(period_config, Dict):
                for k, v in self._period_default.items():
                    if k not in period_config:
                        period_config[k] = v

        # Validate period input
        if isinstance(period_config, int):
            if len(data) < period_config:
                raise ValueError(
                    "len(data) must be greater than, or equal to the period."
                )
        elif isinstance(period_config, dict):
            for [key, period] in period_config.items():
                if len(data) < period:
                    raise ValueError(
                        f"len(data) must be greater than, or equal to each period. \n[Key={key}, Period={period}, Data_Len={len(data)}]"
                    )

        self._data = data
        self._period_config = period_config
        self._memory = memory
        self._retention = retention
        self._init = init

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

    def set_data(self, data: pd.DataFrame):
        self._data = data

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
