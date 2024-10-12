import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volume import MFI


def test_mfi(mfi_df: pd.DataFrame, evaluate: Eval):
    expected = mfi_df["mfi"].to_numpy(dtype=np.float64)
    rolling = MFI(mfi_df).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "MFI")


def test_mfi_update(mfi_df: pd.DataFrame, evaluate: Eval):
    rolling = MFI(mfi_df.iloc[:20])

    for _, series in mfi_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        mfi_df["mfi"].to_numpy(dtype=np.float64),
        rolling.to_numpy(dtype=np.float64),
        "MFI_UPDATE",
    )
