import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from tests.logging import log

from rolling_ta.trend import ADX


def test_adx(adx: ADX, adx_df: pd.DataFrame, evaluate: Eval):
    adx.fit(data=adx_df)
    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        adx.calc().to_numpy(dtype=np.float64),
        "ADX",
    )


def test_adx_update(adx: ADX, adx_df: pd.DataFrame, evaluate: Eval):
    adx.fit(data=adx_df.iloc[:50])
    adx.calc()

    for _, series in adx_df.iloc[50:].iterrows():
        adx.update(series)

    evaluate(
        adx_df["adx"].to_numpy(dtype=np.float64),
        adx.to_numpy(dtype=np.float64),
        "ADX_UPDATE",
    )
