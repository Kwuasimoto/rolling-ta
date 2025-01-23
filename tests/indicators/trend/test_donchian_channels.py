import numpy as np
import pandas as pd

from rolling_ta.trend import HMA
from rolling_ta.trend.donchian_channels import DonchianChannels
from tests.fixtures.eval import Eval


def test_donchian_channels_high(
    donchian_channels: DonchianChannels,
    donchian_channels_df: pd.DataFrame,
    evaluate: Eval,
):
    donchian_channels.fit(donchian_channels_df)
    evaluate(
        donchian_channels_df["highs"].to_numpy(dtype=np.float64),
        donchian_channels.calc().to_numpy("high"),
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
        donchian_channels.calc().to_numpy("low"),
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
