import numpy as np
import pandas as pd

from rolling_ta.trend import HMA
from rolling_ta.trend.donchian_channels import DonchianChannels
from tests.fixtures.eval import Eval


def test_donchian_channels_high(donchian_channels_df: pd.DataFrame, evaluate: Eval):
    expected = donchian_channels_df["highs"].to_numpy(dtype=np.float64)
    rolling = DonchianChannels(donchian_channels_df).to_numpy("high")
    evaluate(expected, rolling, "donchian_highs")


def test_donchian_channels_low(donchian_channels_df: pd.DataFrame, evaluate: Eval):
    expected = donchian_channels_df["lows"].to_numpy(dtype=np.float64)
    rolling = DonchianChannels(donchian_channels_df).to_numpy("low")
    evaluate(expected, rolling, "donchian_lows")


def test_donchian_channels_center(donchian_channels_df: pd.DataFrame, evaluate: Eval):
    expected = donchian_channels_df["centers"].to_numpy(dtype=np.float64)
    rolling = DonchianChannels(donchian_channels_df).to_numpy("center")
    evaluate(expected, rolling, "donchian_centers")
