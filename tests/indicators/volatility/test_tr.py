from logging import Logger
import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volatility import TR


def test_tr(atr_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        atr_df["tr"].to_numpy(dtype=np.float64),
        TR(atr_df).tr().to_numpy(dtype=np.float64),
        name="NUMBA_TR",
    )


def test_tr_update(atr_df: pd.DataFrame, evaluate: Eval):
    rolling = TR(atr_df.iloc[:40])

    for _, series in atr_df.iloc[40:].iterrows():
        rolling.update(series)

    evaluate(
        atr_df["tr"].to_numpy(dtype=np.float64),
        rolling.tr().to_numpy(dtype=np.float64),
        name="NUMBA_TR_UPDATE",
    )
