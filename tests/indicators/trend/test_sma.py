import numpy as np
import pandas as pd

from rolling_ta.trend import SMA
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_sma(sma: SMA, sma_df: pd.DataFrame, evaluate: Eval):
    sma.fit(data=sma_df)
    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        sma.calc().to_numpy(dtype=np.float64),
        "SMA",
    )


def test_sma_update(sma: SMA, sma_df: pd.DataFrame, evaluate: Eval):
    sma.fit(data=sma_df.iloc[:20])
    sma.calc()

    for _, series in sma_df.iloc[20:].iterrows():
        sma.update(series)

    evaluate(
        sma_df["sma"].to_numpy(dtype=np.float64),
        sma.to_numpy(dtype=np.float64),
        "SMA_UPDATE",
    )


def test_sma_to_series(sma: SMA, sma_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(sma, sma_df, "sma")


def test_sma_dataframe(
    sma: SMA, sma_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(sma, sma_df, ["sma_14"])


def test_sma_drop_values(sma: SMA, sma_df: pd.DataFrame):
    sma.fit(sma_df)
    sma.calc()
    sma.drop_values()
    assert not hasattr(sma, "_sma"), f"Failed to delete SMA._sma attribute. {sma._sma}"


def test_sma_set_initialized(sma: SMA, sma_df: pd.DataFrame):
    sma.fit(sma_df)
    sma.calc(initialization_state=True)
    sma.set_initialized(state=False)
    assert (
        not sma._initialized
    ), f"Failed to set SMA._initialized to false. {sma._initialized}"
