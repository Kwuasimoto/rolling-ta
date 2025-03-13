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


def test_wma_update(wma: WMA, wma_df: pd.DataFrame, evaluate: Eval):
    wma.fit(data=wma_df[:50])
    wma.calc()

    for _, ohlcv in wma_df.iloc[50:].iterrows():
        wma.update(ohlcv)

    evaluate(
        wma_df["wma"].to_numpy(dtype=np.float64).round(6),
        wma.to_numpy().round(6),
        "WMA_UPDATE",
    )


def test_wma_to_series(wma: WMA, wma_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(wma, wma_df, "wma")


def test_wma_to_dataframe(
    wma: WMA, wma_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(wma, wma_df, ["wma_14"])


def test_wma_drop_values(wma: WMA, wma_df: pd.DataFrame):
    wma.fit(wma_df)
    wma.calc()
    wma.drop_values()
    assert not hasattr(wma, "_wma"), f"Failed to delete WMA._wma attribute. {wma._wma}"


def test_wma_set_initialized(wma: WMA, wma_df: pd.DataFrame):
    wma.fit(wma_df)
    wma.calc(initialization_state=True)
    wma.set_initialized(state=False)
    assert (
        not wma._initialized
    ), f"Failed to set WMA._initialized to false. {wma._initialized}"
