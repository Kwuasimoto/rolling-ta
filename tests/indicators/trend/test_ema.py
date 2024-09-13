import pandas as pd

from rolling_ta.trend import EMA

from tests.fixtures.eval import Eval


def test_sma(ema_df: pd.DataFrame, evaluate: Eval):
    ema_df["ema"] = ema_df["ema"].astype("float64").round(4)

    rolling = EMA(ema_df, period_config=10)

    expected = ema_df["ema"].dropna()
    rolling_ema = rolling.ema().dropna().round(4)

    evaluate(expected, rolling_ema)


def test_sma_update(ema_df: pd.DataFrame, evaluate: Eval):
    rolling = EMA(ema_df.iloc[:20], period_config=10)
    expected = ema_df["ema"].astype("float64").dropna().round(4)

    for _, series in ema_df.iloc[20:].iterrows():
        rolling.update(series)

    rolling_ema = rolling.ema().dropna().round(4)
    evaluate(expected, rolling_ema)
