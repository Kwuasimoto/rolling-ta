import pytest

from rolling_ta.data import XLSLoader, CSVLoader


@pytest.fixture(name="btc_df")
def btc_df(csv_loader: CSVLoader):
    try:
        return csv_loader.read_resource().copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


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


# Used in TrueRange tests.
@pytest.fixture(name="atr_df")
def atr_df(xls_loader: XLSLoader):
    try:
        return xls_loader.read_resource(
            "cs-atr.xlsx",
            columns=[
                "date",
                "high",
                "low",
                "close",
                "h-l",
                "|h-c_p|",
                "|l-c_p|",
                "tr",
                "atr",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="adx_df")
def adx_df(xls_loader: XLSLoader):
    try:
        return xls_loader.read_resource(
            "cs-adx.xlsx",
            columns=[
                "date",
                "high",
                "low",
                "close",
                "tr",
                "+dm",
                "-dm",
                "tr14",
                "+dm14",
                "-dm14",
                "+di14",
                "-di14",
                "dx",
                "adx",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))
        
