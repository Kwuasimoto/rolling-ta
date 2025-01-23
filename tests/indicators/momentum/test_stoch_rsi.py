import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval

from rolling_ta.momentum import StochasticRSI


def test_stoch_k(stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame, evaluate: Eval):
    stoch_rsi.fit(rsi_df)
    evaluate(
        rsi_df["stoch_k"].to_numpy(dtype=np.float64),
        stoch_rsi.calc().to_numpy(get="k"),
        "STOCH K",
    )


def test_stoch_d(stoch_rsi: StochasticRSI, rsi_df: pd.DataFrame, evaluate: Eval):
    stoch_rsi.fit(rsi_df)
    evaluate(
        rsi_df["stoch_d"].to_numpy(dtype=np.float64),
        stoch_rsi.calc().to_numpy(get="d"),
        "STOCH D",
    )
