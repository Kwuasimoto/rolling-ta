import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.trend import EMA


def test_ema(ema: EMA, ema_df: pd.DataFrame, evaluate: Eval):
    ema.fit(data=ema_df)
    evaluate(
        ema_df["ema"].to_numpy(dtype=np.float64),
        ema.calc().to_numpy(dtype=np.float64),
        "EMA",
    )


def test_ema_update(ema: EMA, ema_df: pd.DataFrame, evaluate: Eval):
    ema.fit(data=ema_df.iloc[:20])
    ema.calc()

    for _, series in ema_df.iloc[20:].iterrows():
        ema.update(series)

    evaluate(
        ema_df["ema"].to_numpy(dtype=np.float64),
        ema.to_numpy(dtype=np.float64),
        name="NUMBA_EMA_UPDATE",
    )
