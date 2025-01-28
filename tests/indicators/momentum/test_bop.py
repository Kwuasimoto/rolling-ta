import numpy as np
import pandas as pd

from rolling_ta.momentum import BOP
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_bop(bop: BOP, bop_df: pd.DataFrame, evaluate: Eval):
    bop.fit(data=bop_df, period_config={"bop": 0})
    evaluate(
        bop_df["bop"].to_numpy(dtype=np.float64),
        bop.calc().to_numpy(dtype=np.float64),
        "BOP",
    )


def test_bop_smoothed(bop: BOP, bop_df: pd.DataFrame, evaluate: Eval):
    bop.fit(bop_df, period_config={"bop": 14})
    evaluate(
        bop_df["bop_14"].to_numpy(dtype=np.float64),
        bop.calc().to_numpy(dtype=np.float64),
        "BOP",
    )


def test_bop_to_series(bop: BOP, bop_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(bop, bop_df, "bop")


def test_bop_to_dataframe(
    bop: BOP, bop_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(bop, bop_df, ["bop_14"])
