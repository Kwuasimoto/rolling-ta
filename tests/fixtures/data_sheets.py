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


@pytest.fixture(name="wma_df")
def wma_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-wma.xlsx",
            columns=[
                "ts",
                "close",
                "weights",
                "weighted_sum",
                "wma",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="hma_df")
def hma_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-hma.xlsx",
            columns=[
                "ts",
                "close",
                "weights",
                "wma_sqrt_sum",
                "wma_sqrt",
                "wma_half_sum",
                "wma_half",
                "wma_full_sum",
                "wma_full",
                "hma_raw",
                "hma_raw_sum",
                "hma",
            ],
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
                "rsi_min_14",
                "rsi_max_14",
                "stoch_rsi",
                "stoch_k",
                "stoch_d",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="bb_df")
def bb_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-bb.xlsx",
            columns=[
                "ts",
                "close",
                "sma",
                "upper",
                "lower",
            ],
        ).copy()

    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="bop_df")
def bop_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-bop.xlsx",
            columns=[
                "ts",
                "open",
                "high",
                "low",
                "close",
                "bop",
                "bop_14",
            ],
        ).copy()

    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="donchian_channels_df")
def donchian_channels_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-donchian.xlsx",
            columns=[
                "ts",
                "high",
                "low",
                "highs",
                "lows",
                "centers",
            ],
        ).copy()

    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


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


@pytest.fixture(name="lr_df")
def lr_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-linear_regression.xlsx",
            columns=[
                "ts",
                "high",
                "low",
                "close",
                "typical",
                "row",
                "intercepts",
                "slopes",
                "r2",
                "forecast",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="mfi_df")
def mfi_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-mfi.xlsx",
            columns=[
                "ts",
                "high",
                "low",
                "close",
                "typical",
                "volume",
                "rmf",
                "pmf",
                "nmf",
                "pmf_sum_14",
                "nmf_sum_14",
                "mfi",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))


@pytest.fixture(name="vwap_df")
def vwap_df(xls_loader: XLSXLoader):
    try:
        return xls_loader.read_resource(
            "btc-vwap.xlsx",
            columns=[
                "timestamp",
                "timestamp_mod",
                "high",
                "low",
                "close",
                "typical",
                "volume",
                "raw_accum",
                "vol_accum",
                "vwap",
            ],
        ).copy()
    except FileNotFoundError as fnfe:
        pytest.fail(str(fnfe))
