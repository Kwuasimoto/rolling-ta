import numpy as np
import pandas as pd

from rolling_ta.trend import WMA
from tests.fixtures.eval import Eval


def test_wma(wma: WMA, wma_df: pd.DataFrame, evaluate: Eval):
    wma.fit(wma_df)
    evaluate(
        wma_df["wma"].to_numpy(dtype=np.float64),
        wma.calc().to_numpy(),
        "WMA",
    )
