import numpy as np
import pandas as pd

from rolling_ta.trend import NumbaDMI
from rolling_ta.logging import logger
from tests.fixtures.eval import Eval


def test_numba_dmi(adx_df: pd.DataFrame, evaluate: Eval):

    dmi = NumbaDMI(adx_df)
    expected_pdmi = adx_df["+di14"].to_numpy(np.float64).round(3)
    rolling_pdmi = dmi.pdmi().to_numpy(np.float64).round(3)

    expected_ndmi = adx_df["-di14"].to_numpy(np.float64).round(3)
    rolling_ndmi = dmi.ndmi().to_numpy(np.float64).round(3)

    evaluate(expected_pdmi, rolling_pdmi)
    evaluate(expected_ndmi, rolling_ndmi)


def test_numba_dmi_update(adx_df: pd.DataFrame, evaluate: Eval):

    dmi = NumbaDMI(adx_df.iloc[:20])

    for _, series in adx_df.iloc[20:].iterrows():
        dmi.update(series)

    expected_pdmi = adx_df["+di14"].to_numpy(np.float64).round(3)
    rolling_pdmi = dmi.pdmi().to_numpy(np.float64).round(3)

    evaluate(expected_pdmi, rolling_pdmi)
