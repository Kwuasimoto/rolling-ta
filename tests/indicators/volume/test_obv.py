import numpy as np
import pandas as pd

from rolling_ta.volume import OBV
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_obv(obv: OBV, obv_df: pd.DataFrame, evaluate: Eval):
    obv.fit(data=obv_df)
    evaluate(
        obv_df["obv"].to_numpy(dtype=np.float64),
        obv.calc().to_numpy(dtype=np.float64),
        "OBV",
    )


def test_obv_update(obv: OBV, obv_df: pd.DataFrame, evaluate: Eval):
    obv.fit(data=obv_df.iloc[:20])
    obv.calc()

    for _, series in obv_df.iloc[20:].iterrows():
        obv.update(series)

    evaluate(
        obv_df["obv"].to_numpy(dtype=np.float64),
        obv.to_numpy(dtype=np.float64),
    )


def test_obv_to_series(obv: OBV, obv_df: pd.DataFrame, validate_series: ValidateSeries):
    validate_series(obv, obv_df, "obv")


def test_obv_to_dataframe(
    obv: OBV, obv_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(obv, obv_df, ["obv_14"])
