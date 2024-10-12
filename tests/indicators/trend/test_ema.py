import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import EMA


def test_ema(ema_df: pd.DataFrame, evaluate: Eval):
    expected = ema_df["ema"].to_numpy(dtype=np.float64)
    rolling = EMA(ema_df).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "EMA")


def test_ema_update(ema_df: pd.DataFrame, evaluate: Eval):
    rolling = EMA(ema_df.iloc[:20])

    for _, series in ema_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        ema_df["ema"].to_numpy(dtype=np.float64),
        rolling.to_numpy(dtype=np.float64),
        name="NUMBA_EMA_UPDATE",
    )
