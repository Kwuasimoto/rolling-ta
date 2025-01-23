import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import DMI


def test_pdmi(dmi: DMI, adx_df: pd.DataFrame, evaluate: Eval):
    dmi.fit(adx_df)
    evaluate(
        adx_df["+dmi"].to_numpy(dtype=np.float64),
        dmi.calc().to_numpy(get="pdmi", dtype=np.float64),
        "PDMI",
    )


def test_pdmi_update(dmi: DMI, adx_df: pd.DataFrame, evaluate: Eval):
    dmi.fit(adx_df.iloc[:20])
    dmi.calc()

    for _, series in adx_df.iloc[20:].iterrows():
        dmi.update(series)

    evaluate(
        adx_df["+dmi"].to_numpy(dtype=np.float64),
        dmi.to_numpy(get="pdmi", dtype=np.float64),
        name="PDMI_UPDATE",
    )


def test_ndmi(dmi: DMI, adx_df: pd.DataFrame, evaluate: Eval):
    dmi.fit(adx_df)
    evaluate(
        adx_df["-dmi"].to_numpy(dtype=np.float64),
        dmi.calc().to_numpy(get="ndmi", dtype=np.float64),
        "NDMI",
    )


def test_ndmi_update(dmi: DMI, adx_df: pd.DataFrame, evaluate: Eval):
    dmi.fit(adx_df.iloc[:20])
    dmi.calc()

    for _, series in adx_df.iloc[20:].iterrows():
        dmi.update(series)

    evaluate(
        adx_df["-dmi"].to_numpy(dtype=np.float64),
        dmi.to_numpy(get="ndmi", dtype=np.float64),
        name="NDMI_UPDATE",
    )
