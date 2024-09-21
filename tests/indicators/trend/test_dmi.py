import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import DMI


def test_dmi_pos(adx_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        adx_df["+dmi"].to_numpy(dtype=np.float64),
        DMI(adx_df).pdmi().to_numpy(dtype=np.float64),
        name="NUMBA_DMI_POS",
    )


def test_dmi_pos_update(adx_df: pd.DataFrame, evaluate: Eval):
    rolling = DMI(adx_df.iloc[:20])

    for _, series in adx_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        adx_df["+dmi"].to_numpy(dtype=np.float64),
        rolling.pdmi().to_numpy(dtype=np.float64),
        name="NUMBA_DMI_POS_UPDATE",
    )


def test_dmi_neg(adx_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        adx_df["-dmi"].to_numpy(dtype=np.float64),
        DMI(adx_df).ndmi().to_numpy(dtype=np.float64),
        name="NUMBA_DMI_NEG",
    )


def test_dmi_neg_update(adx_df: pd.DataFrame, evaluate: Eval):
    rolling = DMI(adx_df.iloc[:20])

    for _, series in adx_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        adx_df["-dmi"].to_numpy(dtype=np.float64),
        rolling.ndmi().to_numpy(dtype=np.float64),
        name="NUMBA_DMI_NEG_UPDATE",
    )
