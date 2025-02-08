import numpy as np
import pandas as pd

from rolling_ta.trend import ADX
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_adx(adx: ADX, adx_df: pd.DataFrame, evaluate: Eval):
    adx.fit(data=adx_df)
    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        adx.calc().to_numpy(dtype=np.float64),
        "ADX",
    )


def test_adx_update(adx: ADX, adx_df: pd.DataFrame, evaluate: Eval):
    adx.fit(data=adx_df.iloc[:50])
    adx.calc()

    for _, series in adx_df.iloc[50:].iterrows():
        adx.update(series)

    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        adx.to_numpy(dtype=np.float64),
        "ADX_UPDATE",
    )


def test_adx_dx_to_series(
    adx: ADX,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(adx, adx_df, "dx")


def test_adx_adx_to_series(
    adx: ADX,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(adx, adx_df, "adx")


def test_adx_pdmi_to_series(
    adx: ADX,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(adx, adx_df, "pdmi")


def test_adx_ndmi_to_series(
    adx: ADX,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(adx, adx_df, "ndmi")


def test_adx_tr_to_series(
    adx: ADX,
    adx_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(adx, adx_df, "tr")


def test_adx_to_dataframe(
    adx: ADX, adx_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(adx, adx_df, ["adx_14", "dx_14", "pdmi_14", "ndmi_14", "tr_14"])


def test_adx_drop_values(adx: ADX, adx_df: pd.DataFrame):
    adx.fit(adx_df)
    adx.calc()
    adx.drop_values()
    assert not hasattr(adx, "_adx"), f"Failed to drop ADX._adx {adx._adx}"
    assert not hasattr(adx, "_dx"), f"Failed to drop ADX._dx {adx._dx}"
    assert not hasattr(
        adx._dmi, "_pdmi"
    ), f"Failed to drop ADX.DMI._pdmi {adx._dmi._pdmi}"
    assert not hasattr(
        adx._dmi, "_ndmi"
    ), f"Failed to drop ADX.DMI._ndmi {adx._dmi._ndmi}"
    assert not hasattr(
        adx._dmi._tr, "_tr"
    ), f"Failed to drop ADX.DMI.TR._tr {adx._dmi._tr._tr}"


def test_adx_set_initialized(adx: ADX, adx_df: pd.DataFrame):
    adx.fit(adx_df)
    adx.calc()
    adx.set_initialized(state=False)
    assert not adx._initialized, f"Failed to set ADX._initialized"
    assert not adx._dmi._initialized, f"Failed to set ADX.DMI._initialized"
    assert not adx._dmi._tr._initialized, f"Failed to set ADX.DMI.TR._initialized"
