import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.momentum import RSI


def test_rsi(rsi_df: pd.DataFrame, evaluate: Eval):
    expected = rsi_df["rsi"].to_numpy(dtype=np.float64)
    rolling = RSI(rsi_df).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "RSI")


def test_rsi_update(rsi_df: pd.DataFrame, evaluate: Eval):
    rolling = RSI(rsi_df.iloc[:20])

    for _, series in rsi_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        rsi_df["rsi"].to_numpy(dtype=np.float64),
        rolling.to_numpy(dtype=np.float64),
        "RSI_UPDATE",
    )
