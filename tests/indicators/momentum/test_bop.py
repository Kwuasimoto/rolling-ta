import numpy as np
import pandas as pd

from rolling_ta.momentum import BOP
from tests.fixtures.eval import Eval


def test_bop(bop_df: pd.DataFrame, evaluate: Eval):
    expected = bop_df["bop"].to_numpy(dtype=np.float64)
    rolling = BOP(data=bop_df, init=True, period_config=0).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "BOP")


def test_bop_smoothed(bop_df: pd.DataFrame, evaluate: Eval):
    expected = bop_df["bop_14"].to_numpy(dtype=np.float64)
    rolling = BOP(data=bop_df, init=True).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "BOP")
