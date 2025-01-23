import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volume import VWAP


def test_vwap(vwap: VWAP, vwap_df: pd.DataFrame, evaluate: Eval):
    vwap.fit(data=vwap_df)
    evaluate(
        vwap_df["vwap"].to_numpy(dtype=np.float64),
        vwap.calc().to_numpy(dtype=np.float64),
        name="VWAP",
    )
