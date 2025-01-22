import pandas as pd

from rolling_ta.trend.ema import EMA


def test_indicator_make(btc_df: pd.DataFrame):
    test_indicator = EMA()
    test_indicator.fit(btc_df)
