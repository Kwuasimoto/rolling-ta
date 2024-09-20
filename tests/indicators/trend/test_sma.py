from logging import Logger
import numpy as np
import pandas as pd

from rolling_ta.trend import SMA, NumbaSMA

from tests.fixtures.eval import Eval


def test_sma(sma_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        SMA(sma_df).sma(),
    )


def test_sma_update(sma_df: pd.DataFrame, evaluate: Eval):
    expected = sma_df["sma"]
    rolling = SMA(sma_df.iloc[:20])

    for _, series in sma_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64),
        rolling.sma().to_numpy(dtype=np.float64),
    )


def test_numba_sma(sma_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        NumbaSMA(sma_df).sma().to_numpy(dtype=np.float64),
    )


def test_numba_sma_update(sma_df: pd.DataFrame, evaluate: Eval):
    expected = sma_df["sma"]
    rolling = NumbaSMA(sma_df.iloc[:20])

    for _, series in sma_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64),
        rolling.sma().to_numpy(dtype=np.float64),
    )
