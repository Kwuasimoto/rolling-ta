import numpy as np
import pandas as pd

from rolling_ta.volatility import TrueRange
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_tr(true_range: TrueRange, atr_df: pd.DataFrame, evaluate: Eval):
    true_range.fit(data=atr_df)
    evaluate(
        atr_df["tr"].to_numpy(dtype=np.float64),
        true_range.calc().to_numpy(dtype=np.float64),
        "TR",
    )


def test_tr_update(true_range: TrueRange, atr_df: pd.DataFrame, evaluate: Eval):
    true_range.fit(data=atr_df.iloc[:40])
    true_range.calc()

    for _, series in atr_df.iloc[40:].iterrows():
        true_range.update(series)

    evaluate(
        atr_df["tr"].to_numpy(dtype=np.float64),
        true_range.to_numpy(dtype=np.float64),
        "TR_UPDATE",
    )


def test_tr_to_series(
    true_range: TrueRange, atr_df: pd.DataFrame, validate_series: ValidateSeries
):
    validate_series(true_range, atr_df, "tr")


def test_tr_to_dataframe(
    true_range: TrueRange, atr_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(true_range, atr_df, ["tr_14"])
