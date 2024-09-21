import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.omni import IchimokuCloud


def test_ichimoku_cloud(ichimoku_cloud_df: pd.DataFrame, evaluate: Eval):

    rolling = IchimokuCloud(ichimoku_cloud_df)

    evaluate(
        ichimoku_cloud_df["tenkan"].to_numpy(dtype=np.float64),
        rolling.tenkan().to_numpy(dtype=np.float64),
        name="NUMBA_TENKAN",
    )
    evaluate(
        ichimoku_cloud_df["kijun"].to_numpy(dtype=np.float64),
        rolling.kijun().to_numpy(dtype=np.float64),
        name="NUMBA_KIJUN",
    )
    evaluate(
        ichimoku_cloud_df["senkou_a"].to_numpy(dtype=np.float64),
        rolling.senkou_a().to_numpy(dtype=np.float64),
        name="NUMBA_SENKOU_A",
    )
    evaluate(
        ichimoku_cloud_df["senkou_b"].to_numpy(dtype=np.float64),
        rolling.senkou_b().to_numpy(dtype=np.float64),
        name="NUMBA_SENKOU_B",
    )


def test_ichimoku_cloud_update(ichimoku_cloud_df: pd.DataFrame, evaluate: Eval):

    rolling = IchimokuCloud(ichimoku_cloud_df.iloc[:100])

    for _, series in ichimoku_cloud_df.iloc[100:].iterrows():
        rolling.update(series)

    evaluate(
        ichimoku_cloud_df["tenkan"].to_numpy(dtype=np.float64),
        rolling.tenkan().to_numpy(dtype=np.float64),
        name="NUMBA_TENKAN_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["kijun"].to_numpy(dtype=np.float64),
        rolling.kijun().to_numpy(dtype=np.float64),
        name="NUMBA_KIJUN_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["senkou_a"].to_numpy(dtype=np.float64),
        rolling.senkou_a().to_numpy(dtype=np.float64),
        name="NUMBA_SENKOU_A_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["senkou_b"].to_numpy(dtype=np.float64),
        rolling.senkou_b().to_numpy(dtype=np.float64),
        name="NUMBA_SENKOU_B_UPDATE",
    )
