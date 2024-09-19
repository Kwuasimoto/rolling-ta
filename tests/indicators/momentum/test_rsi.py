import os
import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.logging import logger
from rolling_ta.momentum import NumbaRSI


def test_numba_rsi(rsi_df: pd.DataFrame, evaluate: Eval):
    expected = rsi_df["rsi"].to_numpy(np.float64).round(4)

    rolling = NumbaRSI(rsi_df)
    rolling_rsi = rolling.rsi().to_numpy(np.float64).round(4)

    evaluate(expected, rolling_rsi)


def test_numba_rsi_update(rsi_df: pd.DataFrame, evaluate: Eval):
    expected = rsi_df["rsi"].to_numpy(np.float64).round(4)

    slice_a = rsi_df.iloc[:20]
    slice_b = rsi_df.iloc[20:]

    rolling = NumbaRSI(slice_a)

    for _, series in slice_b.iterrows():
        rolling.update(series)

    rolling_rsi = rolling.rsi().to_numpy(np.float64).round(4)

    evaluate(expected, rolling_rsi)
