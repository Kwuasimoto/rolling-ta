import numpy as np
import pandas as pd

from rolling_ta.trend import NumbaADX
from rolling_ta.logging import logger

from ta.trend import ADXIndicator
from tests.fixtures.eval import Eval


# These tests confirm if the NumbaDMI Indicator is working as expected.


def test_numba_adx(adx_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        NumbaADX(adx_df).adx().to_numpy(dtype=np.float64),
        name="NUMBA_ADX",
    )


def test_numba_adx_update(adx_df: pd.DataFrame, evaluate: Eval):
    rolling = NumbaADX(adx_df.iloc[:50])

    for _, series in adx_df.iloc[50:].iterrows():
        rolling.update(series)

    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        rolling.adx().to_numpy(dtype=np.float64),
        name="NUMBA_ADX_UPDATE",
    )
