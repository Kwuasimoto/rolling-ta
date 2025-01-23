import pytest

from rolling_ta.momentum.bop import BOP
from rolling_ta.momentum.rsi import RSI
from rolling_ta.momentum.stochastic_rsi import StochasticRSI
from rolling_ta.omni.ichimoku_cloud import IchimokuCloud
from rolling_ta.trend.adx import ADX
from rolling_ta.trend.dmi import DMI
from rolling_ta.trend.donchian_channels import DonchianChannels
from rolling_ta.trend.lr import LinearRegression
from rolling_ta.trend.lrf import LinearRegressionForecast
from rolling_ta.trend.lrr import LinearRegressionR2
from rolling_ta.trend.ema import EMA
from rolling_ta.trend.hma import HMA
from rolling_ta.trend.sma import SMA
from rolling_ta.trend.wma import WMA
from rolling_ta.volatility.atr import AverageTrueRange
from rolling_ta.volatility.bb import BollingerBands
from rolling_ta.volatility.tr import TrueRange
from rolling_ta.volume.mfi import MFI
from rolling_ta.volume.obv import OBV
from rolling_ta.volume.vwap import VWAP


@pytest.fixture(autouse=True, name="bop")
def bop():
    return BOP()


@pytest.fixture(autouse=True, name="rsi")
def rsi():
    return RSI()


@pytest.fixture(autouse=True, name="stoch_rsi")
def stoch_rsi():
    return StochasticRSI()


@pytest.fixture(autouse=True, name="ichimoku")
def ichimoku():
    return IchimokuCloud()


@pytest.fixture(autouse=True, name="adx")
def adx():
    return ADX()


@pytest.fixture(autouse=True, name="dmi")
def dmi():
    return DMI()


@pytest.fixture(autouse=True, name="donchian_channels")
def donchian_channels():
    return DonchianChannels()


@pytest.fixture(autouse=True, name="ema")
def ema():
    return EMA()


@pytest.fixture(autouse=True, name="hma")
def hma():
    return HMA()


@pytest.fixture(autouse=True, name="lr")
def lr():
    return LinearRegression()


@pytest.fixture(autouse=True, name="lr2")
def lr2():
    return LinearRegressionR2()


@pytest.fixture(autouse=True, name="lrf")
def lrf():
    return LinearRegressionForecast()


@pytest.fixture(autouse=True, name="sma")
def sma():
    return SMA()


@pytest.fixture(autouse=True, name="wma")
def wma():
    return WMA()


@pytest.fixture(autouse=True, name="atr")
def atr():
    return AverageTrueRange()


@pytest.fixture(autouse=True, name="bollinger_bands")
def bollinger_bands():
    return BollingerBands()


@pytest.fixture(autouse=True, name="true_range")
def true_range():
    return TrueRange()


@pytest.fixture(autouse=True, name="mfi")
def mfi():
    return MFI()


@pytest.fixture(autouse=True, name="obv")
def obv():
    return OBV()


@pytest.fixture(autouse=True, name="vwap")
def vwap():
    return VWAP()
