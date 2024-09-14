from logging import Logger
import numpy as np
import pandas as pd

from rolling_ta.trend import SMA, NumbaSMA

from tests.fixtures.eval import Eval


def test_sma(sma_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        sma_df["sma"].astype("float64").fillna(0).round(4),
        SMA(sma_df, period_config=10).sma().round(4),
    )


def test_sma_update(sma_df: pd.DataFrame, evaluate: Eval):
    expected = sma_df["sma"].astype("float64").fillna(0).round(4)
    rolling = SMA(sma_df.iloc[:20], period_config=10)

    for _, series in sma_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.sma().round(4))


def test_numba_sma(sma_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        sma_df["sma"].astype("float64").fillna(0).round(4),
        NumbaSMA(sma_df, period_config=10).sma().round(4),
    )


def test_numba_sma_update(sma_df: pd.DataFrame, evaluate: Eval):
    expected = sma_df["sma"].astype("float64").fillna(0).round(4)
    rolling = NumbaSMA(sma_df.iloc[:20], period_config=10)

    for _, series in sma_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.sma().round(4))
