import numpy as np
import pandas as pd

from rolling_ta.volatility import BollingerBands
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


def test_bb_ma(bollinger_bands: BollingerBands, bb_df: pd.DataFrame, evaluate: Eval):
    bollinger_bands.fit(data=bb_df)
    evaluate(
        bb_df["sma"].to_numpy(dtype=np.float64).round(6),
        bollinger_bands.calc().to_numpy(dtype=np.float64).round(6),
        name="BB_SMA",
    )


def test_bb_upper(bollinger_bands: BollingerBands, bb_df: pd.DataFrame, evaluate: Eval):
    bollinger_bands.fit(data=bb_df)
    evaluate(
        bb_df["upper"].to_numpy(dtype=np.float64).round(6),
        bollinger_bands.calc().to_numpy(get="upper", dtype=np.float64).round(6),
        name="BB_UPPER",
    )


def test_bb_lower(bollinger_bands: BollingerBands, bb_df: pd.DataFrame, evaluate: Eval):
    bollinger_bands.fit(data=bb_df)
    evaluate(
        bb_df["lower"].to_numpy(dtype=np.float64).round(6),
        bollinger_bands.calc().to_numpy(get="lower", dtype=np.float64).round(6),
        name="BB_LOWER",
    )


def test_bb_ma_to_series(
    bollinger_bands: BollingerBands,
    bb_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(bollinger_bands, bb_df, "ma")


def test_bb_upper_to_series(
    bollinger_bands: BollingerBands,
    bb_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(bollinger_bands, bb_df, "upper")


def test_bb_lower_to_series(
    bollinger_bands: BollingerBands,
    bb_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(bollinger_bands, bb_df, "lower")


def test_bb_to_dataframe(
    bollinger_bands: BollingerBands,
    bb_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    validate_dataframe(bollinger_bands, bb_df, ["ma_20", "upper_20", "lower_20"])


def test_bb_drop_values(bollinger_bands: BollingerBands, bb_df: pd.DataFrame):
    bollinger_bands.fit(data=bb_df)
    bollinger_bands.calc()
    bollinger_bands.drop_values()
    assert not hasattr(bollinger_bands, "_upper")
    assert not hasattr(bollinger_bands._ma, "_ema")
    assert not hasattr(bollinger_bands, "_lower")


def test_bb_set_initialized(bollinger_bands: BollingerBands, bb_df: pd.DataFrame):
    bollinger_bands.fit(data=bb_df)
    bollinger_bands.calc()
    bollinger_bands.set_initialized(state=False)
    assert (
        not bollinger_bands._initialized
    ), f"Failed to set BollingerBands._initialized. state={bollinger_bands.initialized()}"
    assert (
        not bollinger_bands._ma._initialized
    ), f"Failed to set BollingerBands.EMA._initialized. state={bollinger_bands._ma.initialized()}"
