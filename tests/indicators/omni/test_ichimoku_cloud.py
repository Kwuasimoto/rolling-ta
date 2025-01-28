import numpy as np
import pandas as pd

from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame
from rolling_ta.omni import IchimokuCloud


def test_ichimoku_cloud(
    ichimoku: IchimokuCloud, ichimoku_cloud_df: pd.DataFrame, evaluate: Eval
):
    ichimoku.fit(ichimoku_cloud_df)
    ichimoku.calc()

    evaluate(
        ichimoku_cloud_df["tenkan"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="tenkan", dtype=np.float64),
        "TENKAN",
    )
    evaluate(
        ichimoku_cloud_df["kijun"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="kijun", dtype=np.float64),
        "KIJUN",
    )
    evaluate(
        ichimoku_cloud_df["senkou_b"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="senkou_b", dtype=np.float64),
        "SENKOU_B",
    )
    evaluate(
        ichimoku_cloud_df["senkou_a"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="senkou_a", dtype=np.float64),
        "SENKOU_A",
    )


def test_ichimoku_cloud_update(
    ichimoku: IchimokuCloud, ichimoku_cloud_df: pd.DataFrame, evaluate: Eval
):
    ichimoku.fit(ichimoku_cloud_df.iloc[:100])
    ichimoku.calc()

    for _, series in ichimoku_cloud_df.iloc[100:].iterrows():
        ichimoku.update(series)

    evaluate(
        ichimoku_cloud_df["tenkan"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="tenkan", dtype=np.float64),
        "TENKAN_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["kijun"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="kijun", dtype=np.float64),
        "KIJUN_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["senkou_b"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="senkou_b", dtype=np.float64),
        "SENKOU_B_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["senkou_a"].to_numpy(dtype=np.float64),
        ichimoku.to_numpy(get="senkou_a", dtype=np.float64),
        "SENKOU_A_UPDATE",
    )


def test_ichimoku_cloud_tenkan_to_series(
    ichimoku: IchimokuCloud,
    ichimoku_cloud_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(ichimoku, ichimoku_cloud_df, "tenkan")


def test_ichimoku_cloud_kijun_to_series(
    ichimoku: IchimokuCloud,
    ichimoku_cloud_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(ichimoku, ichimoku_cloud_df, "kijun")


def test_ichimoku_cloud_senkou_a_to_series(
    ichimoku: IchimokuCloud,
    ichimoku_cloud_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(ichimoku, ichimoku_cloud_df, "senkou_a")


def test_ichimoku_cloud_senkou_b_to_series(
    ichimoku: IchimokuCloud,
    ichimoku_cloud_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(ichimoku, ichimoku_cloud_df, "senkou_b")


def test_ichimoku_cloud_to_dataframe(
    ichimoku: IchimokuCloud,
    ichimoku_cloud_df: pd.DataFrame,
    validate_dataframe: ValidateDataFrame,
):
    validate_dataframe(
        ichimoku,
        ichimoku_cloud_df,
        ["tenkan_9", "kijun_26", "senkou_b_52", "senkou_a_35"],
    )
