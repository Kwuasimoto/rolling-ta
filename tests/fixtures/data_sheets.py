import importlib.resources as pkg

import pandas as pd
import pytest

from logging import Logger

from rolling_ta.data import XLSLoader


@pytest.fixture(name="obv_df")
def obv_df(xls_loader: XLSLoader):
    try:
        return xls_loader.read_resource(
            "cs-obv.xlsx",
            columns=["date", "close", "up-down", "volume", "pos-neg", "obv"],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))
