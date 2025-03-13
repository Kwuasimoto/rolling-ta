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


def test_hma_update(hma: HMA, hma_df: pd.DataFrame, evaluate: Eval):
    hma.fit(data=hma_df[:50])
    hma.calc()

    for _, ohlcv in hma_df.iloc[50:].iterrows():
        hma.update(ohlcv)

    evaluate(
        hma_df["hma"].to_numpy(dtype=np.float64).round(6),
        hma.to_numpy().round(6),
        "HMA",
    )


def test_hma_to_series(hma: HMA, hma_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(hma, hma_df, "hma")


def test_hma_to_dataframe(
    hma: HMA, hma_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(hma, hma_df, ["hma_14", "wma_full_14", "wma_half_7"])


def test_hma_drop_values(hma: HMA, hma_df: pd.DataFrame):
    hma.fit(hma_df)
    hma.calc()
    hma.drop_values()
    assert not hasattr(hma, "_hma"), f"Failed to delete HMA._hma attribute. {hma._hma}"


def test_hma_set_initialized(hma: HMA, hma_df: pd.DataFrame):
    hma.fit(hma_df)
    hma.calc(initialization_state=True)
    hma.set_initialized(state=False)
    assert (
        not hma._initialized
    ), f"Failed to set HMA._initialized to false. {hma._initialized}"
