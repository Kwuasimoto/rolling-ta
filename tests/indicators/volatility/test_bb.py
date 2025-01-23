import numpy as np
import pandas as pd

from tests.fixtures.eval import Eval
from rolling_ta.volatility import BollingerBands


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
