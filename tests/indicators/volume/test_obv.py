import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volume import OBV


def test_obv(obv_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        obv_df["obv"].to_numpy(dtype=np.float64),
        OBV(obv_df).obv().to_numpy(dtype=np.float64),
    )


def test_obv_update(obv_df: pd.DataFrame, evaluate: Eval):
    expected = obv_df["obv"]
    rolling = OBV(obv_df.iloc[:20])

    for _, series in obv_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        expected.to_numpy(dtype=np.float64),
        rolling.obv().to_numpy(dtype=np.float64),
    )
