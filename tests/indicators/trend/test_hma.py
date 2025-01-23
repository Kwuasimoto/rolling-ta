import numpy as np
import pandas as pd

from rolling_ta.trend import HMA
from tests.fixtures.eval import Eval


def test_hma(hma: HMA, hma_df: pd.DataFrame, evaluate: Eval):
    hma.fit(hma_df)
    evaluate(
        hma_df["hma"].to_numpy(dtype=np.float64).round(6),
        hma.calc().to_numpy().round(6),
        "HMA",
    )
