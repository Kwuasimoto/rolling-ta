import numpy as np
import pandas as pd

from rolling_ta.trend import EMA
from tests.fixtures.helpers import Eval, ValidateSeries, ValidateDataFrame


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


def test_ema_to_series(
    ema: EMA,
    ema_df: pd.DataFrame,
    validate_series: ValidateSeries,
):
    validate_series(ema, ema_df, "ema")


def test_ema_to_dataframe(
    ema: EMA, ema_df: pd.DataFrame, validate_dataframe: ValidateDataFrame
):
    validate_dataframe(ema, ema_df, ["ema_14"])


def test_ema_drop_values(ema: EMA, ema_df: pd.DataFrame):
    ema.fit(ema_df)
    ema.calc()
    ema.drop_values()
    assert not hasattr(ema, "_ema"), f"Failed to delete EMA._ema attribute. {ema._ema}"


def test_ema_set_initialized(ema: EMA, ema_df: pd.DataFrame):
    ema.fit(ema_df)
    ema.calc(initialization_state=True)
    ema.set_initialized(state=False)
    assert (
        not ema._initialized
    ), f"Failed to set EMA._initialized to false. {ema._initialized}"
