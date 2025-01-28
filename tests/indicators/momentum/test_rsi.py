import numpy as np
import pandas as pd

from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


from rolling_ta.momentum import RSI


def test_rsi(rsi: RSI, rsi_df: pd.DataFrame, evaluate: Eval):
    rsi.fit(rsi_df)
    evaluate(
        rsi_df["rsi"].to_numpy(dtype=np.float64),
        rsi.calc().to_numpy(dtype=np.float64),
        "RSI",
    )


def test_rsi_update(rsi: RSI, rsi_df: pd.DataFrame, evaluate: Eval):
    rsi.fit(rsi_df.iloc[:20])
    rsi.calc()

    for _, series in rsi_df.iloc[20:].iterrows():
        rsi.update(series)

    evaluate(
        rsi_df["rsi"].to_numpy(dtype=np.float64),
        rsi.to_numpy(dtype=np.float64),
        "RSI_UPDATE",
    )


def test_rsi_to_series(rsi: RSI, rsi_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(rsi, rsi_df, "rsi")


def test_rsi_to_dataframe(
    rsi: RSI, rsi_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(rsi, rsi_df, ["rsi_14"])
