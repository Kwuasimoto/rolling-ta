import numpy as np
import pandas as pd

from rolling_ta.volume import VWAP
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_vwap(vwap: VWAP, vwap_df: pd.DataFrame, evaluate: Eval):
    vwap.fit(data=vwap_df)
    evaluate(
        vwap_df["vwap"].to_numpy(dtype=np.float64),
        vwap.calc().to_numpy(dtype=np.float64),
        name="VWAP",
    )


def test_vwap_to_series(
    vwap: VWAP, vwap_df: pd.DataFrame, validate_series: ValidateSeries
):
    validate_series(vwap, vwap_df, "vwap")


def test_vwap_to_dataframe(
    vwap: VWAP, vwap_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(vwap, vwap_df, ["vwap_1440"])
