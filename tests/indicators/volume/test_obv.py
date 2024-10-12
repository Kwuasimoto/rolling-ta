import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volume import OBV


def test_obv(obv_df: pd.DataFrame, evaluate: Eval):
    expected = obv_df["obv"].to_numpy(dtype=np.float64)
    rolling = OBV(obv_df).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "OBV")


def test_obv_update(obv_df: pd.DataFrame, evaluate: Eval):
    expected = obv_df["obv"]
    rolling = OBV(obv_df.iloc[:20])

    for _, series in obv_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64),
        rolling.to_numpy(dtype=np.float64),
    )
