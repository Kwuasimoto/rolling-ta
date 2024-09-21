import numpy as np
import pandas as pd

from rolling_ta.indicator import Indicator
from rolling_ta.logging import logger
from rolling_ta.trend import SMA

from typing import Optional


class BollingerBands(Indicator):
    """
    Bollinger Bands Indicator.

    Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations) which are used to
    identify volatility and potential overbought or oversold conditions in an asset.

    Material
    --------
        https://www.investopedia.com/terms/b/bollingerbands.asp
        https://pypi.org/project/ta/
    """

    pass
