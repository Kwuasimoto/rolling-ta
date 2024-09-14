from logging import Logger
import numpy as np
import pandas as pd

from rolling_ta.volatility import TrueRange, NumbaTrueRange

from tests.fixtures.eval import Eval


def test_tr(atr_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        atr_df["tr"].astype("float64").fillna(0).round(4),
        TrueRange(atr_df).tr().round(4),
    )


def test_tr_update(atr_df: pd.DataFrame, evaluate: Eval):
    expected = atr_df["tr"].astype("float64").fillna(0).round(4)
    rolling = TrueRange(atr_df.iloc[:20])

    for _, series in atr_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.tr().round(4))


def test_numba_tr(atr_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        atr_df["tr"].astype("float64").fillna(0).round(4),
        NumbaTrueRange(atr_df).tr().round(4),
    )


def test_numba_tr_update(atr_df: pd.DataFrame, evaluate: Eval):
    expected = atr_df["tr"].astype("float64").fillna(0).round(4)
    rolling = NumbaTrueRange(atr_df.iloc[:20])

    for _, series in atr_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.tr().round(4))
