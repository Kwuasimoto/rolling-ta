from numpy import mean
import pandas as pd

from rolling_ta.omni import NumbaIchimokuCloud
from rolling_ta.logging import logger

from ta.trend import IchimokuIndicator
from tests.fixtures.eval import Eval


# These tests confirm if the NumbaDMI Indicator is working as expected.

# If someone can explain the following:
# How is my init test working (test_numba_ichimoku_cloud) when comparing line b,
# But when I perform the rolling test (test_numba_ichimoku_cloud_update)


# Initialization tests
def test_numba_ichimoku_cloud(btc_df: pd.DataFrame, evaluate: Eval):
    data = btc_df.iloc[:200].copy()

    rolling = NumbaIchimokuCloud(data)
    expected = IchimokuIndicator(data["high"], data["low"])

    a_start = max(rolling.period("tenkan"), rolling.period("kijun")) - 1
    b_start = rolling.period("senkou") - 1

    rolling_senkou_a = rolling.senkou_a()[a_start:]
    rolling_senkou_b = rolling.senkou_b()[b_start:]
    expected_senkou_a = expected.ichimoku_a().iloc[a_start:]
    expected_senkou_b = expected.ichimoku_b().iloc[b_start:]

    evaluate(
        expected.ichimoku_conversion_line().fillna(0).round(3),
        rolling.tenkan().round(3),
    )
    evaluate(
        expected.ichimoku_base_line().fillna(0).round(3),
        rolling.kijun().round(3),
    )
    evaluate(
        expected_senkou_a.fillna(0).round(3),
        rolling_senkou_a.round(3),
    )
    evaluate(
        expected_senkou_b.fillna(0).round(3),
        rolling_senkou_b.round(3),
    )


# def test_numba_ichimoku_cloud_update(btc_df: pd.DataFrame, evaluate: Eval):
#     data = btc_df.iloc[:200].copy()

#     expected = IchimokuIndicator(data["high"], data["low"])
#     rolling = NumbaIchimokuCloud(data.iloc[:100])

#     for _, series in data.iloc[100:].iterrows():
#         rolling.update(series)

#     a_start = max(rolling.period("tenkan"), rolling.period("kijun")) - 1
#     b_start = rolling.period("senkou") - 1

#     rolling_senkou_a = rolling.senkou_a()[a_start:]
#     rolling_senkou_b = rolling.senkou_b()[b_start:]
#     expected_senkou_a = expected.ichimoku_a().iloc[a_start:]
#     expected_senkou_b = expected.ichimoku_b().iloc[b_start:]

#     expected.ichimoku_base_line()
#     expected.ichimoku_conversion_line()
#     expected.ichimoku_a()
#     expected.ichimoku_b()

#     evaluate(
#         expected.ichimoku_conversion_line().fillna(0).round(3),
#         rolling.tenkan().round(3),
#     )
#     evaluate(
#         expected.ichimoku_base_line().fillna(0).round(3),
#         rolling.kijun().round(3),
#     )
#     evaluate(
#         expected_senkou_a.fillna(0).round(3),
#         rolling_senkou_a.round(3),
#     )
#     evaluate(
#         expected_senkou_b.fillna(0).round(3),
#         rolling_senkou_b.round(3),
#     )
