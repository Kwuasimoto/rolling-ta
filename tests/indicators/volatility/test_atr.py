from logging import Logger
import numpy as np
import pandas as pd

from rolling_ta.volatility import AverageTrueRange, NumbaAverageTrueRange

from tests.fixtures.eval import Eval


def test_atr(atr_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        atr_df["atr"].astype("float64").fillna(0).round(4),
        AverageTrueRange(atr_df).atr().round(4),
    )


def test_atr_update(atr_df: pd.DataFrame, evaluate: Eval):
    expected = atr_df["atr"].astype("float64").fillna(0).round(4)
    rolling = AverageTrueRange(atr_df.iloc[:20])

    for _, series in atr_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.atr().round(4))


def test_numba_atr(atr_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        atr_df["atr"].astype("float64").fillna(0).round(4),
        NumbaAverageTrueRange(atr_df, period_config=14).atr().round(4),
    )


def test_numba_atr_update(atr_df: pd.DataFrame, evaluate: Eval):
    expected = atr_df["atr"].astype("float64").fillna(0).round(4)
    rolling = NumbaAverageTrueRange(atr_df.iloc[:20])

    for _, series in atr_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.atr().round(4))
