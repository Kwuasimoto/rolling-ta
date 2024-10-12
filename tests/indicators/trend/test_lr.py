import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval

from rolling_ta.trend import (
    LinearRegression,
    LinearRegressionR2,
    LinearRegressionForecast,
)


def test_intercepts(lr_df: pd.DataFrame, evaluate: Eval):
    expected = lr_df["intercepts"].to_numpy(dtype=np.float64)
    rolling = LinearRegression(lr_df).to_numpy(get="intercept")
    evaluate(expected, rolling, "intercepts")


def test_slopes(lr_df: pd.DataFrame, evaluate: Eval):
    expected = lr_df["slopes"].to_numpy(dtype=np.float64)
    rolling = LinearRegression(lr_df).to_numpy(get="slope")
    evaluate(expected, rolling, "slopes")


def test_r2(lr_df: pd.DataFrame, evaluate: Eval):
    expected = lr_df["r2"].to_numpy(dtype=np.float64)
    rolling = LinearRegressionR2(lr_df).to_numpy(get="r2")
    evaluate(expected, rolling, "r2")


def test_forecast(lr_df: pd.DataFrame, evaluate: Eval):
    expected = lr_df["forecast"].to_numpy(dtype=np.float64)
    rolling = LinearRegressionForecast(lr_df, period_config={"lrf": 0}).to_numpy(
        get="forecast"
    )
    evaluate(expected, rolling, "forecast")
