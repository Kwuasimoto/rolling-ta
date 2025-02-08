from typing import Callable, Optional
import numpy as np
import pandas as pd
import pytest

from rolling_ta.indicator import Indicator
from rolling_ta.volatility.bb import BollingerBands
from tests.logging import log


Eval = Callable[[np.ndarray[np.float64], np.ndarray[np.float64], Optional[str]], None]


@pytest.fixture(name="evaluate")
def evaluate():
    def e(
        expected: np.ndarray[np.float64],
        rolling: np.ndarray[np.float64],
        name: Optional[str] = "unnamed",
    ):
        if len(expected) != len(rolling):
            pytest.fail(
                f"Length equivalency: [loc={name}, expected={len(expected)}, rolling={len(rolling)}]"
            )
            raise Exception("STOP")

        for i, [e, r] in enumerate(zip(expected, rolling)):
            if not np.isclose(e, r, atol=1e-6):
                log.error(f"Equals: [loc={name}, index={i}, expected={e}, rolling={r}]")
                log.error(
                    f"Equals: [loc={name}, expected=\n{expected}\n, rolling=\n{rolling}\n]"
                )
                pytest.fail(
                    f"Equals: [loc={name}, index={i}, expected={e}, rolling={r}]"
                )
                raise Exception("STOP")

    return e


ValidateSeries = Callable[[Indicator, pd.DataFrame, str], None]


def handle_validate_series_generics(indicator: Indicator, series: pd.Series):
    """Reassigns the series name of lower level indicator results until a programmable solution is found."""
    if type(indicator) is BollingerBands:
        if series.name == "sma_20":
            mod = f"ma_{indicator._period_config['ma']}"
            log.debug(f"Modding {indicator.__class__.__name__} series name -> {mod}")
            series.name = mod


@pytest.fixture(name="validate_series")
def validate_series():
    def e(indi: Indicator, data: pd.DataFrame, key: str):
        indi.fit(data=data)
        indi.calc()
        series = indi.to_series(key)

        # Handle unique cases like bollinger where a generic ma can be used for "ma" key.
        handle_validate_series_generics(indi, series)
        log.debug(f"[Comparing series names]")
        assert (
            series.name == f"{key}_{indi.period(key)}"
        ), "@compare failed to assign column to indicator series output properly."

    return e


ValidateDataFrame = Callable[[Indicator, pd.DataFrame, list[str]], None]


@pytest.fixture(name="validate_dataframe")
def validate_dataframe():
    def f(indi: Indicator, data: pd.DataFrame, columns: list[str]):
        indi.fit(data, indi._period_config)
        indi.calc()
        df = indi.to_dataframe()
        key_periods = []

        for key in indi._keys:
            log.debug(f"Check if [key={key}] in [period_config={indi._period_config}]")
            # If key in period config, create the period_key column name.
            if key in indi._period_config:
                key_period = f"{key}_{indi._period_config[key]}"
                log.debug(f"Created key_period [key_period={key_period}]")

                # There are some indicators which periods do not directly translate to an array of values.
                # ex: stoch_k is actually stoch_rsi., and stoch_k is just a weight.
                log.debug(f"Check if [key_period={key_period}] in {columns}")
                if key_period in columns:
                    key_periods.append(key_period)

        df_column_key_periods = []
        for column in df.columns:
            key_period = column.split("_", maxsplit=1)[-1]
            df_column_key_periods.append(key_period)

        assert len(df), "to_dataframe returned an empty dataframe"
        assert (
            set(columns) == set(key_periods) == set(df_column_key_periods)
        ), f"columns does not match keys: [columns={columns}, keys={key_periods}, df.cols={df.columns.values}]"

    return f
