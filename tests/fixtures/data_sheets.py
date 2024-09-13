import pytest

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


@pytest.fixture(name="sma_df")
def sma_df(xls_loader: XLSLoader):
    try:
        return xls_loader.read_resource(
            "cs-sma.xlsx", columns=["date", "close", "sma"]
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="ema_df")
def ema_df(xls_loader: XLSLoader):
    try:
        return xls_loader.read_resource(
            "cs-ema.xlsx", columns=["date", "close", "ema"]
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))
