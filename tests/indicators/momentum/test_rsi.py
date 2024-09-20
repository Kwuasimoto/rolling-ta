import os
import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.logging import logger
from rolling_ta.momentum import NumbaRSI


def test_numba_rsi(rsi_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        rsi_df["rsi"].to_numpy(dtype=np.float64).round(2),
        NumbaRSI(rsi_df).rsi().to_numpy(dtype=np.float64).round(2),
    )


def test_numba_rsi_update(rsi_df: pd.DataFrame, evaluate: Eval):
    expected = rsi_df["rsi"]

    rolling = NumbaRSI(rsi_df.iloc[:20])

    for _, series in rsi_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64).round(2),
        rolling.rsi().to_numpy(dtype=np.float64).round(2),
    )
