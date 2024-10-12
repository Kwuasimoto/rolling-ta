import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volatility import BollingerBands


def test_bb_ma(bb_df: pd.DataFrame, evaluate: Eval):
    expected = bb_df["sma"].to_numpy(dtype=np.float64).round(6)
    rolling = BollingerBands(bb_df).to_numpy(dtype=np.float64).round(6)
    evaluate(expected, rolling, name="BB_SMA")


def test_bb_upper(bb_df: pd.DataFrame, evaluate: Eval):
    expected = bb_df["upper"].to_numpy(dtype=np.float64).round(6)
    rolling = BollingerBands(bb_df).to_numpy(get="upper", dtype=np.float64).round(6)
    evaluate(expected, rolling, name="BB_UPPER")


def test_bb_lower(bb_df: pd.DataFrame, evaluate: Eval):
    expected = bb_df["lower"].to_numpy(dtype=np.float64).round(6)
    rolling = BollingerBands(bb_df).to_numpy(get="lower", dtype=np.float64).round(6)
    evaluate(expected, rolling, name="BB_LOWER")
