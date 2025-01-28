import numpy as np
import pandas as pd

from rolling_ta.trend import WMA
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_wma(wma: WMA, wma_df: pd.DataFrame, evaluate: Eval):
    wma.fit(wma_df)
    evaluate(
        wma_df["wma"].to_numpy(dtype=np.float64),
        wma.calc().to_numpy(),
        "WMA",
    )


def test_wma_to_series(wma: WMA, wma_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(wma, wma_df, "wma")


def test_wma_to_dataframe(
    wma: WMA, wma_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(wma, wma_df, ["wma_14"])
