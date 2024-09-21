import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import ADX


def test_adx(adx_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        ADX(adx_df).adx().to_numpy(dtype=np.float64),
        name="NUMBA_ADX",
    )


def test_adx_update(adx_df: pd.DataFrame, evaluate: Eval):
    rolling = ADX(adx_df.iloc[:50])

    for _, series in adx_df.iloc[50:].iterrows():
        rolling.update(series)

    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        rolling.adx().to_numpy(dtype=np.float64),
        name="NUMBA_ADX_UPDATE",
    )
