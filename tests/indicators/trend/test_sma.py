from logging import Logger
import numpy as np
import pandas as pd

from rolling_ta.trend import SMA

from tests.fixtures.eval import Eval


def test_sma(sma_df: pd.DataFrame, evaluate: Eval):
    sma_df["sma"] = sma_df["sma"].astype("float64").round(4)

    rolling = SMA(sma_df, period_config=10)

    expected = sma_df["sma"].dropna()
    rolling_sma = rolling.sma().dropna().round(4)

    evaluate(expected, rolling_sma)


def test_sma_update(sma_df: pd.DataFrame, evaluate: Eval):
    rolling = SMA(sma_df.iloc[:20], period_config=10)
    expected = sma_df["sma"].astype("float64").dropna().round(4)

    for _, series in sma_df.iloc[20:].iterrows():
        rolling.update(series)

    rolling_sma = rolling.sma().dropna().round(4)
    evaluate(expected, rolling_sma)
