import pandas as pd
import numpy as np
from typing import Union, List, Dict


class Indicator:

    _data: pd.DataFrame
    _period_config: Union[int, Dict[str, int]]
    _memory: bool
    _retention: Union[int, None]
    _init: bool
    _count = 0

    def __init__(
        self,
        data: pd.DataFrame,
        period_config: Union[int, Dict[str, int]],
        memory: bool,
        retention: Union[int, None],
        init: bool,
    ) -> None:
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
        else:
            raise ValueError("Invalid type for periods. Must be int, or dict.")

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

    def init(self):
        pass

    def update(self, data: pd.Series):
        pass

    def apply_retention(self):
        pass

    def drop_data(self):
        self._data = None
