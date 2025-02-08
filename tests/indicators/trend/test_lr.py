import numpy as np
import pandas as pd


from rolling_ta.trend import (
    LinearRegression,
    LinearRegressionR2,
    LinearRegressionForecast,
)
from tests.logging import log
from tests.fixtures.helpers import (
    Eval,
    ValidateSeries,
    ValidateDataFrame,
)


def test_intercepts(lr: LinearRegression, lr_df: pd.DataFrame, evaluate: Eval):
    log.debug(f"{lr.get_config()}")
    lr.fit(data=lr_df)
    evaluate(
        lr_df["intercept"].to_numpy(dtype=np.float64),
        lr.calc().to_numpy(get="intercept"),
        "intercepts",
    )


def test_slopes(lr: LinearRegression, lr_df: pd.DataFrame, evaluate: Eval):
    log.debug(f"{lr.get_config()}")
    lr.fit(data=lr_df)
    evaluate(
        lr_df["slope"].to_numpy(dtype=np.float64),
        lr.calc().to_numpy(get="slope"),
        "slopes",
    )


def test_r2(lr2: LinearRegressionR2, lr_df: pd.DataFrame, evaluate: Eval):
    log.debug(f"{lr2.get_config()}")
    lr2.fit(data=lr_df)
    evaluate(
        lr_df["lr2"].to_numpy(dtype=np.float64),
        lr2.calc().to_numpy(get="lr2"),
        "lr2",
    )


def test_forecast(lrf: LinearRegressionForecast, lr_df: pd.DataFrame, evaluate: Eval):
    log.debug(f"{lrf.get_config()}")
    lrf.fit(data=lr_df, period_config={"price": 14, "lrf": 0})
    evaluate(
        lr_df["forecast"].to_numpy(dtype=np.float64),
        lrf.calc().to_numpy(get="lrf"),
        "forecast",
    )


def test_lr_intercepts_to_series(
    lr: LinearRegression,
    lr_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(lr, lr_df, "intercept")


def test_lr_slopes_to_series(
    lr: LinearRegression,
    lr_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(lr, lr_df, "slope")


def test_lr_to_dataframe(
    lr: LinearRegression,
    lr_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    validate_dataframe(lr, lr_df, ["price_14", "slope_14", "intercept_14"])


def test_lr_drop_values(
    lr: LinearRegression,
    lr_df: pd.DataFrame,
):
    lr.fit(lr_df)
    lr.calc()
    lr.drop_values()
    assert not hasattr(
        lr, "_slope"
    ), f"Failed to drop LinearRegression._slope. {lr._slope}"
    assert not hasattr(
        lr, "_intercept"
    ), f"Failed to drop LinearRegression._intercept. {lr._intercept}"


def test_lr_set_initialized(
    lr: LinearRegression,
    lr_df: pd.DataFrame,
):
    lr.fit(lr_df)
    lr.calc()
    lr.set_initialized(state=False)
    assert (
        not lr._initialized
    ), f"Failed to set LinearRegression._initialized. {lr._initialized}"


def test_lr2_to_series(
    lr2: LinearRegressionR2,
    lr_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(lr2, lr_df, "lr2")


def test_lr2_to_dataframe(
    lr2: LinearRegressionR2,
    lr_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    validate_dataframe(lr2, lr_df, ["lr2_14", "price_14", "slope_14", "intercept_14"])


def test_lr2_drop_values(
    lr2: LinearRegressionR2,
    lr_df: pd.DataFrame,
):
    lr2.fit(lr_df)
    lr2.calc()
    lr2.drop_values()
    assert not hasattr(lr2, "_lr2")
    assert not hasattr(lr2._lr, "_slope")
    assert not hasattr(lr2._lr, "_intercept")


def test_lr2_set_initialized(
    lr2: LinearRegressionR2,
    lr_df: pd.DataFrame,
):
    lr2.fit(lr_df)
    lr2.calc()
    lr2.set_initialized(state=False)
    assert not lr2._initialized
    assert not lr2._lr._initialized


def test_lrf_to_series(
    lrf: LinearRegressionForecast,
    lr_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(lrf, lr_df, "lrf")


def test_lrf_to_dataframe(
    lrf: LinearRegressionForecast,
    lr_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    lrf.set_period_config({"lrf": 0, "price": 14, "slope": 14, "intercept": 14})
    validate_dataframe(lrf, lr_df, ["lrf_0", "price_14", "slope_14", "intercept_14"])


def test_lrf_drop_values(
    lrf: LinearRegressionForecast,
    lr_df: pd.DataFrame,
):
    lrf.fit(lr_df)
    lrf.calc()
    lrf.drop_values()
    assert not hasattr(lrf, "_lrf")
    assert not hasattr(lrf._lr, "_slope")
    assert not hasattr(lrf._lr, "_intercept")


def test_lrf_set_initialized(
    lrf: LinearRegressionForecast,
    lr_df: pd.DataFrame,
):
    lrf.fit(lr_df)
    lrf.calc()
    lrf.set_initialized(state=False)
    assert not lrf._initialized
    assert not lrf._lr._initialized
