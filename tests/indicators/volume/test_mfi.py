import numpy as np
import pandas as pd
import numba as nb

from rolling_ta.volume.mfi import NumbaMFI
from ta.volume import MFIIndicator

from tests.fixtures.eval import Eval
from rolling_ta.logging import logger


def test_numba_mfi(mfi_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        mfi_df["mfi"].to_numpy(dtype=np.float64),
        NumbaMFI(mfi_df).mfi().to_numpy(dtype=np.float64),
        name="NUMBA_MFI",
    )


# def test_numba_mfi_update(btc_df: pd.DataFrame, evaluate: Eval):
#     expected = btc_df["obv"].astype("float64").fillna(0).round(4)
#     rolling = NumbaMFI(btc_df.iloc[:20])

#     for _, series in btc_df.iloc[20:].iterrows():
#         rolling.update(series)

#     evaluate(expected, rolling.obv())
