import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volatility import TR


def test_tr(atr_df: pd.DataFrame, evaluate: Eval):
    expected = atr_df["tr"].to_numpy(dtype=np.float64)
    rolling = TR(data=atr_df, init=True).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "TR")


def test_tr_update(atr_df: pd.DataFrame, evaluate: Eval):
    rolling = TR(data=atr_df.iloc[:40], init=True)

    for _, series in atr_df.iloc[40:].iterrows():
        rolling.update(series)

    evaluate(
        atr_df["tr"].to_numpy(dtype=np.float64),
        rolling.to_numpy(dtype=np.float64),
        "TR_UPDATE",
    )
