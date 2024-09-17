import numpy as np
import pandas as pd
import numba as nb

from rolling_ta.volume.mfi import NumbaMFI
from ta.volume import MFIIndicator

from tests.fixtures.eval import Eval
from rolling_ta.logging import logger


# def test_numba_mfi(btc_df: pd.DataFrame, evaluate: Eval):
#     expected = MFIIndicator(
#         btc_df["high"], btc_df["low"], btc_df["close"], btc_df["volume"]
#     )
#     expected_mfi = expected.money_flow_index().fillna(0).to_numpy(np.float64).round(4)

#     rolling = NumbaMFI(btc_df)
#     rolling_mfi = rolling.mfi().fillna(0).to_numpy(np.float64).round(4)

#     evaluate(rolling_mfi, expected_mfi)


# def test_numba_mfi_update(btc_df: pd.DataFrame, evaluate: Eval):
#     expected = btc_df["obv"].astype("float64").fillna(0).round(4)
#     rolling = NumbaMFI(btc_df.iloc[:20])

#     for _, series in btc_df.iloc[20:].iterrows():
#         rolling.update(series)

#     evaluate(expected, rolling.obv())
