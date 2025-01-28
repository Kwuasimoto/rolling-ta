from wsgiref import validate
import numpy as np
import pandas as pd

from rolling_ta.volatility import AverageTrueRange
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_atr(atr: AverageTrueRange, atr_df: pd.DataFrame, evaluate: Eval):
    atr.fit(data=atr_df)
    evaluate(
        atr_df["atr"].to_numpy(dtype=np.float64),
        atr.calc().to_numpy(dtype=np.float64),
        "ATR",
    )


def test_atr_update(atr: AverageTrueRange, atr_df: pd.DataFrame, evaluate: Eval):
    atr.fit(data=atr_df.iloc[:20])
    atr.calc()

    for _, series in atr_df.iloc[20:].iterrows():
        atr.update(series)

    evaluate(
        atr_df["atr"].to_numpy(dtype=np.float64),
        atr.to_numpy(dtype=np.float64),
        "ATR_UPDATE",
    )


def test_atr_to_series(
    atr: AverageTrueRange, atr_df: pd.DataFrame, validate_series: ValidateSeries
):
    validate_series(atr, atr_df, "atr")


def test_atr_to_dataframe(
    atr: AverageTrueRange, atr_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(atr, atr_df, ["atr_14", "tr_14"])
