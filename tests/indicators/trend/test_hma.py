import numpy as np
import pandas as pd

from rolling_ta.trend import HMA
from tests.fixtures.eval import Eval


def test_hma(hma_df: pd.DataFrame, evaluate: Eval):
    expected = hma_df["hma"].to_numpy(dtype=np.float64).round(6)
    rolling = HMA(hma_df).to_numpy().round(6)
    evaluate(expected, rolling, "HMA")
