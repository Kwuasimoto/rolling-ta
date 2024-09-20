import pytest

from rolling_ta.data import XLSXLoader, CSVLoader


@pytest.fixture(name="btc_df")
def btc_df(csv_loader: CSVLoader):
    try:
        return csv_loader.read_resource().copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="obv_df")
def obv_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-obv.xlsx",
            columns=["ts", "close", "volume", "up", "down", "obv"],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="sma_df")
def sma_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-sma.xlsx", columns=["ts", "close", "sma"]
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="ema_df")
def ema_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-ema.xlsx", columns=["ts", "close", "ema"]
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="rsi_df")
def rsi_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-rsi.xlsx",
            columns=[
                "ts",
                "close",
                "gain",
                "loss",
                "gain_14",
                "loss_14",
                "rsi",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


# Used in TrueRange tests.
@pytest.fixture(name="atr_df")
def atr_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-atr.xlsx",
            columns=[
                "ts",
                "high",
                "low",
                "close",
                "tr",
                "atr",
            ],
        ).copy()

    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="adx_df")
def adx_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-adx.xlsx",
            columns=[
                "ts",
                "high",
                "low",
                "close",
                "tr",
                "atr",
                "h-h_p",
                "l_p-l",
                "+dx",
                "-dx",
                "+dx_14",
                "-dx_14",
                "+dmi",
                "-dmi",
                "dx",
                "adx",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="ichimoku_cloud_df")
def ichimoku_cloud_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-ichimoku_cloud.xlsx",
            columns=[
                "ts",
                "high",
                "low",
                "high_max_9",
                "low_min_9",
                "high_max_26",
                "low_max_26",
                "high_max_52",
                "low_max_52",
                "tenkan",
                "kijun",
                "senkou_a",
                "senkou_b",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))
