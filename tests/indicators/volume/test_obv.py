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


def test_obv_drop_values(obv: OBV, obv_df: pd.DataFrame):
    obv.fit(obv_df)
    obv.calc()
    obv.drop_values()
    assert not hasattr(obv, "_obv"), f"Failed to delete OBV._obv attribute. {obv._obv}"


def test_obv_set_initialized(obv: OBV, obv_df: pd.DataFrame):
    obv.fit(obv_df)
    obv.calc(initialization_state=True)
    obv.set_initialized(state=False)
    assert (
        not obv._initialized
    ), f"Failed to set OBV._initialized to false. {obv._initialized}"
