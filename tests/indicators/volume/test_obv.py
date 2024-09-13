import pandas as pd
from logging import Logger

from rolling_ta.volume import OBV

from tests.fixtures.eval import Eval


def test_obv(obv_df: pd.DataFrame, evaluate: Eval):
    rolling = OBV(obv_df).obv()
    expected = obv_df["obv"]
    evaluate(expected, rolling, "TEST_OBV")


def test_obv_update(obv_df: pd.DataFrame, evaluate: Eval):
    rolling = OBV(obv_df.iloc[:20])
    expected = obv_df["obv"]

    for _, series in obv_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.obv(), "TEST_OBV_UPDATE")
