import numpy as np
import pandas as pd

from rolling_ta.trend import NumbaADX
from rolling_ta.logging import logger

from ta.trend import ADXIndicator
from tests.fixtures.eval import Eval


# These tests confirm if the NumbaDMI Indicator is working as expected.


def test_numba_adx(btc_df: pd.DataFrame, evaluate: Eval):
    data = btc_df.iloc[:200].copy()

    expected = ADXIndicator(data["high"], data["low"], data["close"])
    rolling = NumbaADX(data)

    evaluate(expected.adx().round(4), rolling.adx().round(4))


def test_numba_adx_update(btc_df: pd.DataFrame, evaluate: Eval):
    data = btc_df.iloc[:200].copy()

    expected = ADXIndicator(data["high"], data["low"], data["close"])
    rolling = NumbaADX(data.iloc[:100])

    for _, series in data.iloc[100:].iterrows():
        rolling.update(series)

    rolling_ema = rolling
    evaluate(
        expected.adx().to_numpy(np.float64).round(4),
        rolling_ema.adx().to_numpy(np.float64).round(4),
    )
