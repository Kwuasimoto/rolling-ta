import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volume import OBV


def test_obv(obv_df: pd.DataFrame, evaluate: Eval):
    expected = obv_df["obv"].to_numpy(dtype=np.float64)
    rolling = OBV(data=obv_df, init=True).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "OBV")


def test_obv_update(obv_df: pd.DataFrame, evaluate: Eval):
    expected = obv_df["obv"]
    rolling = OBV(data=obv_df.iloc[:20], init=True)

    for _, series in obv_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64),
        rolling.to_numpy(dtype=np.float64),
    )
