import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.omni import IchimokuCloud


def test_ichimoku_cloud(ichimoku_cloud_df: pd.DataFrame, evaluate: Eval):

    rolling = IchimokuCloud(ichimoku_cloud_df)

    evaluate(
        ichimoku_cloud_df["tenkan"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="tenkan", dtype=np.float64),
        "TENKAN",
    )
    evaluate(
        ichimoku_cloud_df["kijun"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="kijun", dtype=np.float64),
        "KIJUN",
    )
    evaluate(
        ichimoku_cloud_df["senkou_a"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="senkou_a", dtype=np.float64),
        "SENKOU_A",
    )
    evaluate(
        ichimoku_cloud_df["senkou_b"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="senkou_b", dtype=np.float64),
        "SENKOU_B",
    )


def test_ichimoku_cloud_update(ichimoku_cloud_df: pd.DataFrame, evaluate: Eval):

    rolling = IchimokuCloud(ichimoku_cloud_df.iloc[:100])

    for _, series in ichimoku_cloud_df.iloc[100:].iterrows():
        rolling.update(series)

    evaluate(
        ichimoku_cloud_df["tenkan"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="tenkan", dtype=np.float64),
        "TENKAN_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["kijun"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="kijun", dtype=np.float64),
        "KIJUN_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["senkou_a"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="senkou_a", dtype=np.float64),
        "SENKOU_A_UPDATE",
    )
    evaluate(
        ichimoku_cloud_df["senkou_b"].to_numpy(dtype=np.float64),
        rolling.to_numpy(get="senkou_b", dtype=np.float64),
        "SENKOU_B_UPDATE",
    )
