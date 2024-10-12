import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import DMI


def test_pdmi(adx_df: pd.DataFrame, evaluate: Eval):
    expected = adx_df["+dmi"].to_numpy(dtype=np.float64)
    rolling = DMI(adx_df).to_numpy(get="pdmi", dtype=np.float64)
    evaluate(expected, rolling, "PDMI")


def test_pdmi_update(adx_df: pd.DataFrame, evaluate: Eval):
    rolling = DMI(adx_df.iloc[:20])

    for _, series in adx_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        adx_df["+dmi"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="pdmi", dtype=np.float64),
        name="PDMI_UPDATE",
    )


def test_ndmi(adx_df: pd.DataFrame, evaluate: Eval):
    expected = adx_df["-dmi"].to_numpy(dtype=np.float64)
    rolling = DMI(adx_df).to_numpy(get="ndmi", dtype=np.float64)
    evaluate(expected, rolling, "NDMI")


def test_ndmi_update(adx_df: pd.DataFrame, evaluate: Eval):
    rolling = DMI(adx_df.iloc[:20])

    for _, series in adx_df.iloc[20:].iterrows():
        rolling.update(series)

    evaluate(
        adx_df["-dmi"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="ndmi", dtype=np.float64),
        name="NDMI_UPDATE",
    )
