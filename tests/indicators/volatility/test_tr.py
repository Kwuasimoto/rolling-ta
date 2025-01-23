import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volatility import TrueRange


def test_tr(true_range: TrueRange, atr_df: pd.DataFrame, evaluate: Eval):
    true_range.fit(data=atr_df)
    evaluate(
        atr_df["tr"].to_numpy(dtype=np.float64),
        true_range.calc().to_numpy(dtype=np.float64),
        "TR",
    )


def test_tr_update(true_range: TrueRange, atr_df: pd.DataFrame, evaluate: Eval):
    true_range.fit(data=atr_df.iloc[:40])
    true_range.calc()

    for _, series in atr_df.iloc[40:].iterrows():
        true_range.update(series)

    evaluate(
        atr_df["tr"].to_numpy(dtype=np.float64),
        true_range.to_numpy(dtype=np.float64),
        "TR_UPDATE",
    )
