import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval

from rolling_ta.momentum import StochasticRSI


def test_stoch_k(rsi_df: pd.DataFrame, evaluate: Eval):
    expected = rsi_df["stoch_k"].to_numpy(dtype=np.float64)
    rolling = StochasticRSI(
        rsi_df, period_config={"rsi": 14, "stoch": 10, "k": 3, "d": 3}
    ).to_numpy(get="k")
    evaluate(expected, rolling, "STOCH K")


def test_stoch_d(rsi_df: pd.DataFrame, evaluate: Eval):
    expected = rsi_df["stoch_d"].to_numpy(dtype=np.float64)
    rolling = StochasticRSI(
        rsi_df, period_config={"rsi": 14, "stoch": 10, "k": 3, "d": 3}
    ).to_numpy(get="d")
    evaluate(expected, rolling, "STOCH D")
