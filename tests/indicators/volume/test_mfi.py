import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volume import MFI


def test_mfi(mfi: MFI, mfi_df: pd.DataFrame, evaluate: Eval):
    mfi.fit(data=mfi_df)
    evaluate(
        mfi_df["mfi"].to_numpy(dtype=np.float64),
        mfi.calc().to_numpy(dtype=np.float64),
        "MFI",
    )


def test_mfi_update(mfi: MFI, mfi_df: pd.DataFrame, evaluate: Eval):
    mfi.fit(data=mfi_df.iloc[:20])
    mfi.calc()

    for _, series in mfi_df.iloc[20:].iterrows():
        mfi.update(series)

    evaluate(
        mfi_df["mfi"].to_numpy(dtype=np.float64),
        mfi.to_numpy(dtype=np.float64),
        "MFI_UPDATE",
    )
