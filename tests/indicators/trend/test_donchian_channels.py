import numpy as np
import pandas as pd


from rolling_ta.trend.donchian_channels import DonchianChannels
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_donchian_channels_high(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    evaluate: Eval,
):
    donchian_channels.fit(donchian_channels_df)
    evaluate(
        donchian_channels_df["highs"].to_numpy(dtype=np.float64),
        donchian_channels.calc().to_numpy("highs"),
        "donchian_highs",
    )


def test_donchian_channels_low(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    evaluate: Eval,
):
    donchian_channels.fit(donchian_channels_df)
    evaluate(
        donchian_channels_df["lows"].to_numpy(dtype=np.float64),
        donchian_channels.calc().to_numpy("lows"),
        "donchian_lows",
    )


def test_donchian_channels_center(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    evaluate: Eval,
):
    donchian_channels.fit(donchian_channels_df)
    evaluate(
        donchian_channels_df["centers"].to_numpy(dtype=np.float64),
        donchian_channels.calc().to_numpy("center"),
        "donchian_centers",
    )


def test_donchian_channels_highs_to_series(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(donchian_channels, donchian_channels_df, "highs")


def test_donchian_channels_center_to_series(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(donchian_channels, donchian_channels_df, "center")


def test_donchian_channels_lows_to_series(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(donchian_channels, donchian_channels_df, "lows")


def test_donchian_channels_to_dataframe(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    validate_dataframe(
        donchian_channels, donchian_channels_df, ["highs_14", "center_14", "lows_14"]
    )


def test_donchian_channels_drop_values(
    donchian_channels: DonchianChannels, donchian_channels_df: pd.DataFrame
):
    donchian_channels.fit(donchian_channels_df)
    donchian_channels.calc()
    donchian_channels.drop_values()
    assert not hasattr(
        donchian_channels, "_highs"
    ), f"Failed to delete DonchianChannels._tenkan attribute. {donchian_channels._highs}"
    assert not hasattr(
        donchian_channels, "_center"
    ), f"Failed to delete DonchianChannels._kijun attribute. {donchian_channels._center}"
    assert not hasattr(
        donchian_channels, "_lows"
    ), f"Failed to delete DonchianChannels._senkou_a attribute. {donchian_channels._lows}"


def test_donchian_channels_set_initialized(
    donchian_channels: DonchianChannels, donchian_channels_df: pd.DataFrame
):
    donchian_channels.fit(donchian_channels_df)
    donchian_channels.calc(initialization_state=True)
    donchian_channels.set_initialized(state=False)
    assert (
        not donchian_channels._initialized
    ), f"Failed to set DonchianChannels._initialized to false. {donchian_channels._initialized}"
