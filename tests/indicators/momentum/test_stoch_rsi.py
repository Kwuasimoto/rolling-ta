import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.logging import logger
from rolling_ta.momentum import NumbaStochasticRSI
from ta.momentum import StochRSIIndicator


def test_numba_stoch_rsi(rsi_df: pd.DataFrame, evaluate: Eval):

    expected = StochRSIIndicator(rsi_df["close"])
    expected_stoch = expected.stochrsi().fillna(0).to_numpy(np.float64).round(4)

    rolling = NumbaStochasticRSI(
        rsi_df, period_config={"rsi": 14, "stoch": 14, "k": 3, "d": 3}
    )
    rolling_stoch = rolling.stoch_rsi().to_numpy(np.float64).round(4)

    evaluate(expected_stoch, rolling_stoch)


# def test_numba_stoch_rsi_update(rsi_df: pd.DataFrame, evaluate: Eval):
#     expected = rsi_df["rsi"].to_numpy(np.float64).round(4)

#     slice_a = rsi_df.iloc[:20]
#     slice_b = rsi_df.iloc[20:]

#     rolling = NumbaStochasticRSI(slice_a)

#     for _, series in slice_b.iterrows():
#         rolling.update(series)

#     rolling_rsi = rolling.rsi().to_numpy(np.float64).round(4)

#     evaluate(expected, rolling_rsi)
