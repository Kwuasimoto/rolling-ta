import numpy as np
import pandas as pd

from rolling_ta.trend import WMA
from tests.fixtures.eval import Eval


def test_wma(wma_df: pd.DataFrame, evaluate: Eval):
    expected = wma_df["wma"].to_numpy(dtype=np.float64)
    rolling = WMA(wma_df).to_numpy()
    evaluate(expected, rolling, "WMA")
