import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volatility import AverageTrueRange


def test_atr(atr_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        atr_df["atr"].to_numpy(dtype=np.float64),
        AverageTrueRange(atr_df).atr().to_numpy(dtype=np.float64),
        name="NUMBA_ATR",
    )


def test_atr_update(atr_df: pd.DataFrame, evaluate: Eval):
    rolling = AverageTrueRange(atr_df.iloc[:20])

    for _, series in atr_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        atr_df["atr"].to_numpy(dtype=np.float64),
        rolling.atr().to_numpy(dtype=np.float64),
        name="NUMBA_ATR_UPDATE",
    )
