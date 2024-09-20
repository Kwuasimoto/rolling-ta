import numpy as np
import pandas as pd

from rolling_ta.trend import EMA, NumbaEMA
from rolling_ta.logging import logger

from tests.fixtures.eval import Eval


def test_ema(ema_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        ema_df["ema"].to_numpy(dtype=np.float64),
        EMA(ema_df).ema().to_numpy(dtype=np.float64),
    )


def test_ema_update(ema_df: pd.DataFrame, evaluate: Eval):
    expected = ema_df["ema"]
    rolling = EMA(ema_df.iloc[:20])

    for _, series in ema_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64),
        rolling.ema().to_numpy(dtype=np.float64),
    )


def test_numba_ema(ema_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        ema_df["ema"].to_numpy(dtype=np.float64),
        NumbaEMA(ema_df).ema().to_numpy(dtype=np.float64),
    )


def test_numba_ema_update(ema_df: pd.DataFrame, evaluate: Eval):
    expected = ema_df["ema"]
    rolling = NumbaEMA(ema_df.iloc[:20])

    for _, series in ema_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64),
        rolling.ema().to_numpy(dtype=np.float64),
    )
