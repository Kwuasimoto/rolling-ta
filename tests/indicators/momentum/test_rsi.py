import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
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
