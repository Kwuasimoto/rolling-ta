import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval

from rolling_ta.trend import (
    LinearRegression,
    LinearRegressionR2,
    LinearRegressionForecast,
)


def test_intercepts(lr: LinearRegression, lr_df: pd.DataFrame, evaluate: Eval):
    lr.fit(data=lr_df)
    evaluate(
        lr_df["intercepts"].to_numpy(dtype=np.float64),
        lr.calc().to_numpy(get="intercept"),
        "intercepts",
    )


def test_slopes(lr: LinearRegression, lr_df: pd.DataFrame, evaluate: Eval):
    lr.fit(data=lr_df)
    evaluate(
        lr_df["slopes"].to_numpy(dtype=np.float64),
        lr.calc().to_numpy(get="slope"),
        "slopes",
    )


def test_r2(lr2: LinearRegressionR2, lr_df: pd.DataFrame, evaluate: Eval):
    lr2.fit(data=lr_df)
    evaluate(
        lr_df["r2"].to_numpy(dtype=np.float64),
        lr2.calc().to_numpy(get="r2"),
        "r2",
    )


def test_forecast(lrf: LinearRegressionForecast, lr_df: pd.DataFrame, evaluate: Eval):
    lrf.fit(data=lr_df, period_config={"lr": 14, "lrf": 0})
    evaluate(
        lr_df["forecast"].to_numpy(dtype=np.float64),
        lrf.calc().to_numpy(get="forecast"),
        "forecast",
    )
