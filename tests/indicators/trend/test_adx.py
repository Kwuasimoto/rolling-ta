import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import ADX


def test_adx(adx_df: pd.DataFrame, evaluate: Eval):
    expected = adx_df["adx"].to_numpy(dtype=np.float64)
    rolling = ADX(data=adx_df, init=True).to_numpy(dtype=np.float64)
    evaluate(expected, rolling, "ADX")


def test_adx_update(adx_df: pd.DataFrame, evaluate: Eval):
    rolling = ADX(data=adx_df.iloc[:50], init=True)

    for _, series in adx_df.iloc[50:].iterrows():
        rolling.update(series)

    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        rolling.to_numpy(dtype=np.float64),
        "ADX_UPDATE",
    )
