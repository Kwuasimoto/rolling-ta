from logging import Logger
import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import SMA


def test_sma(sma_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        SMA(sma_df).sma().to_numpy(dtype=np.float64),
        name="NUMBA_SMA",
    )


def test_sma_update(sma_df: pd.DataFrame, evaluate: Eval):
    rolling = SMA(sma_df.iloc[:20])

    for _, series in sma_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        rolling.sma().to_numpy(dtype=np.float64),
        name="NUMBA_SMA_UPDATE",
    )
