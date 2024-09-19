import numpy as np
import pandas as pd

from ta.trend import ADXIndicator

from rolling_ta.trend import NumbaDMI
from rolling_ta.logging import logger
from tests.fixtures.eval import Eval


# Somehow, I managed to properly calculate DMI and period sooner. This is why the tests are spliced.


def test_numba_dmi_pos(btc_df: pd.DataFrame, evaluate: Eval):
    sample = btc_df.iloc[:30]

    expected = ADXIndicator(sample["high"], sample["low"], sample["close"])
    rolling = NumbaDMI(sample)

    expected_pdmi = expected.adx_pos().to_numpy(np.float64).round(4)
    rolling_pdmi = rolling.pdmi().to_numpy(np.float64).round(4)

    evaluate(expected_pdmi[15:], rolling_pdmi[15:])


def test_numba_dmi_pos_update(btc_df: pd.DataFrame, evaluate: Eval):
    sample = btc_df.iloc[:30]

    expected = ADXIndicator(sample["high"], sample["low"], sample["close"])
    rolling = NumbaDMI(sample.iloc[:20])

    for _, series in sample.iloc[20:].iterrows():
        rolling.update(series)

    expected_pdmi = expected.adx_pos().to_numpy(np.float64).round(4)
    rolling_pdmi = rolling.pdmi().to_numpy(np.float64).round(4)

    evaluate(expected_pdmi[15:], rolling_pdmi[15:])


def test_numba_dmi_neg(btc_df: pd.DataFrame, evaluate: Eval):
    sample = btc_df.iloc[:30]

    expected = ADXIndicator(sample["high"], sample["low"], sample["close"])
    rolling = NumbaDMI(sample)

    expected_pdmi = expected.adx_neg().to_numpy(np.float64).round(4)
    rolling_pdmi = rolling.ndmi().to_numpy(np.float64).round(4)

    evaluate(expected_pdmi[15:], rolling_pdmi[15:])


def test_numba_dmi_neg_update(btc_df: pd.DataFrame, evaluate: Eval):
    sample = btc_df.iloc[:30]

    expected = ADXIndicator(sample["high"], sample["low"], sample["close"])
    rolling = NumbaDMI(sample.iloc[:20])

    for _, series in sample.iloc[20:].iterrows():
        rolling.update(series)

    expected_pdmi = expected.adx_neg().to_numpy(np.float64).round(4)
    rolling_pdmi = rolling.ndmi().to_numpy(np.float64).round(4)

    evaluate(expected_pdmi[15:], rolling_pdmi[15:])
