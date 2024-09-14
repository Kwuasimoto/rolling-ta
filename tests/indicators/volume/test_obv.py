import pandas as pd

from rolling_ta.volume import OBV, NumbaOBV

from tests.fixtures.eval import Eval


def test_obv(obv_df: pd.DataFrame, evaluate: Eval):
    evaluate(obv_df["obv"].astype("float64").fillna(0).round(4), OBV(obv_df).obv())


def test_obv_update(obv_df: pd.DataFrame, evaluate: Eval):
    expected = obv_df["obv"].astype("float64").fillna(0).round(4)
    rolling = OBV(obv_df.iloc[:20])

    for _, series in obv_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.obv())


def test_numba_obv(obv_df: pd.DataFrame, evaluate: Eval):
    evaluate(obv_df["obv"].astype("float64").fillna(0).round(4), NumbaOBV(obv_df).obv())


def test_numba_obv_update(obv_df: pd.DataFrame, evaluate: Eval):
    expected = obv_df["obv"].astype("float64").fillna(0).round(4)
    rolling = NumbaOBV(obv_df.iloc[:20])

    for _, series in obv_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(expected, rolling.obv())
