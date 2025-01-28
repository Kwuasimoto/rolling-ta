import numpy as np
import pandas as pd

from rolling_ta.volume import MFI
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


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


def test_mfi_to_series(mfi: MFI, mfi_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(mfi, mfi_df, "mfi")


def test_mfi_to_dataframe(
    mfi: MFI, mfi_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(mfi, mfi_df, ["mfi_14"])
