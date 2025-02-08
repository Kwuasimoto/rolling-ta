import numpy as np
import pandas as pd

from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame
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


def test_dmi_pdmi_to_series(
    dmi: DMI,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(dmi, adx_df, "pdmi")


def test_dmi_ndmi_to_series(
    dmi: DMI,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(dmi, adx_df, "ndmi")


def test_dmi_tr_to_series(
    dmi: DMI,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(dmi, adx_df, "tr")


def test_dmi_to_dataframe(
    dmi: DMI,
    adx_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    validate_dataframe(dmi, adx_df, ["pdmi_14", "ndmi_14", "tr_14"])


def test_dmi_drop_values(dmi: DMI, adx_df: pd.DataFrame):
    dmi.fit(adx_df)
    dmi.calc()
    dmi.drop_values()
    assert not hasattr(dmi, "_pdmi"), f"Failed to drop DMI._pdmi {dmi._pdmi}"
    assert not hasattr(dmi, "_ndmi"), f"Failed to drop DMI._ndmi {dmi._ndmi}"
    assert not hasattr(dmi._tr, "_tr"), f"Failed to drop ADX.DMI._pdmi {dmi._tr._tr}"


def test_adx_set_initialized(dmi: DMI, adx_df: pd.DataFrame):
    dmi.fit(adx_df)
    dmi.calc()
    dmi.set_initialized(state=False)
    assert not dmi._initialized, f"Failed to set DMI._initialized"
    assert not dmi._tr._initialized, f"Failed to set DMI.TR._initialized"
