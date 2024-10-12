import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volume import VWAP


def test_vwap(vwap_df: pd.DataFrame, evaluate: Eval):
    expected = vwap_df["vwap"].to_numpy(dtype=np.float64)
    rolling = VWAP(vwap_df).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, name="VWAP")
