import numpy as np
import pandas as pd

from rolling_ta.momentum import StochasticRSI
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_stoch_k(stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame, evaluate: Eval):
    stoch_rsi.fit(rsi_df)
    evaluate(
        rsi_df["stoch_k"].to_numpy(dtype=np.float64),
        stoch_rsi.calc().to_numpy(get="stoch_rsi"),
        "STOCH K",
    )


def test_stoch_d(stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame, evaluate: Eval):
    stoch_rsi.fit(rsi_df)
    evaluate(
        rsi_df["stoch_d"].to_numpy(dtype=np.float64),
        stoch_rsi.calc().to_numpy(get="stoch_d"),
        "STOCH D",
    )


def test_stoch_k_to_series(
    stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame, validate_series: ValidateSeries
):
    validate_series(stoch_rsi, rsi_df, "stoch_rsi")


def test_stoch_d_to_series(
    stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame, validate_series: ValidateSeries
):
    validate_series(stoch_rsi, rsi_df, "stoch_d")


def test_stoch_rsi_to_dataframe(
    stoch_rsi: StochasticRSI,
    rsi_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    validate_dataframe(stoch_rsi, rsi_df, ["rsi_14", "stoch_rsi_10", "stoch_d_3"])


def test_stoch_rsi_drop_values(stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame):
    stoch_rsi.fit(rsi_df)
    stoch_rsi.calc()
    stoch_rsi.drop_values()
    assert not hasattr(
        stoch_rsi, "_stoch_rsi"
    ), f"Failed to delete StochasticRSI._stoch_rsi attribute. {stoch_rsi._stoch_rsi}"
    assert not hasattr(
        stoch_rsi, "_stoch_d"
    ), f"Failed to delete StochasticRSI._stoch_d attribute. {stoch_rsi._stoch_d}"
    assert not hasattr(
        stoch_rsi._rsi, "_rsi"
    ), f"Failed to delete StochasticRSI.RSI._rsi attribute. {stoch_rsi._rsi}"


def test_stoch_rsi_set_initialized(stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame):
    stoch_rsi.fit(rsi_df)
    stoch_rsi.calc(initialization_state=True)
    stoch_rsi.set_initialized(state=False)
    assert (
        not stoch_rsi._initialized
    ), f"Failed to set StochasticRSI._initialized to false. {stoch_rsi._initialized}"
    assert (
        not stoch_rsi._rsi._initialized
    ), f"Failed to set StochasticRSI.RSI._initialized to false. {stoch_rsi._rsi._initialized}"
