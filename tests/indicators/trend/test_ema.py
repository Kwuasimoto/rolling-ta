import pandas as pd

from rolling_ta.trend import EMA, NumbaEMA
from rolling_ta.logging import logger

from tests.fixtures.eval import Eval


def test_ema(ema_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        ema_df["ema"].astype("float64").fillna(0).round(4),
        EMA(ema_df, period_config=10).ema().round(4),
    )


def test_ema_update(ema_df: pd.DataFrame, evaluate: Eval):
    expected = ema_df["ema"].astype("float64").fillna(0).round(4)
    rolling = EMA(ema_df.iloc[:20], period_config=10)

    for _, series in ema_df.iloc[20:].iterrows():
        rolling.update(series)

    rolling_ema = rolling
    evaluate(expected, rolling_ema.ema().round(4))


def test_numba_ema(ema_df: pd.DataFrame, evaluate: Eval):
    evaluate(
        ema_df["ema"].astype("float64").fillna(0).round(4),
        NumbaEMA(ema_df, period_config=10).ema().round(4),
    )


def test_numba_ema_update(ema_df: pd.DataFrame, evaluate: Eval):
    expected = ema_df["ema"].astype("float64").fillna(0).round(4)
    rolling = NumbaEMA(ema_df.iloc[:20], period_config=10)

    for _, series in ema_df.iloc[20:].iterrows():
        rolling.update(series)

    rolling_ema = rolling
    evaluate(expected, rolling_ema.ema().round(4))
