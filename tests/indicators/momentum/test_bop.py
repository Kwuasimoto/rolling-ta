import numpy as np
import pandas as pd

from rolling_ta.momentum import BOP
from tests.fixtures.eval import Eval


from rolling_ta.logging import log


def test_bop(bop: BOP, bop_df: pd.DataFrame, evaluate: Eval):
    bop.fit(data=bop_df, period_config=0)
    evaluate(
        bop_df["bop"].to_numpy(dtype=np.float64),
        bop.calc().to_numpy(dtype=np.float64),
        "BOP",
    )


def test_bop_smoothed(bop: BOP, bop_df: pd.DataFrame, evaluate: Eval):
    bop.fit(bop_df, period_config=14)
    evaluate(
        bop_df["bop_14"].to_numpy(dtype=np.float64),
        bop.calc().to_numpy(dtype=np.float64),
        "BOP",
    )
