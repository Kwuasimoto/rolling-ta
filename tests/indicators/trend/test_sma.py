import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import SMA


def test_sma(sma: SMA, sma_df: pd.DataFrame, evaluate: Eval):
    sma.fit(data=sma_df)
    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        sma.calc().to_numpy(dtype=np.float64),
        "SMA",
    )


def test_sma_update(sma: SMA, sma_df: pd.DataFrame, evaluate: Eval):
    sma.fit(data=sma_df.iloc[:20])
    sma.calc()

    for _, series in sma_df.iloc[20:].iterrows():
        sma.update(series)

    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        sma.to_numpy(dtype=np.float64),
        "SMA_UPDATE",
    )
