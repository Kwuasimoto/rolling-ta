import numpy as np
import pandas as pd

from rolling_ta.trend import HMA
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_hma(hma: HMA, hma_df: pd.DataFrame, evaluate: Eval):
    hma.fit(hma_df)
    evaluate(
        hma_df["hma"].to_numpy(dtype=np.float64).round(6),
        hma.calc().to_numpy().round(6),
        "HMA",
    )


def test_hma_to_series(hma: HMA, hma_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(hma, hma_df, "hma")


def test_hma_to_dataframe(
    hma: HMA, hma_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(hma, hma_df, ["hma_14", "wma_full_14", "wma_half_7"])
