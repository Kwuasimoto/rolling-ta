# DO NOT EXPOSE TO PACKAGE.

import importlib.resources as pkg

import numpy as np
import pandas as pd

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, IchimokuIndicator

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volume import MFIIndicator


from rolling_ta.data import CSVLoader
from rolling_ta.logging import logger

from rolling_ta.volatility import BollingerBands as BB, AverageTrueRange as ATR
from rolling_ta.trend import EMA, SMA, ADX

from rolling_ta.momentum import RSI, StochasticRSI
from rolling_ta.volume import MFI

from rolling_ta.omni import IchimokuCloud

from time import time

# -- Data Loading Tests --

loader = CSVLoader()

# if __name__ == "__main__":
#     values = loader.read_resource()
#     logger.debug(f"\n{values}\n")


# -- Building Zone --

# if __name__ == "__main__":
#     data = loader.read_resource()
#     copy = data[:28].copy()

#     expected = ATR(copy)

# -- Comparison Tests --


def expMovingAverage(values, window):
    weights = np.exp(np.linspace(-1.0, 0.0, window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[: len(values)]
    a[:window] = a[window]
    return a


# if __name__ == "__main__":
#     data = loader.read_resource("btc_sample_sentdex.csv")
#     copy = data.copy()

#     close = copy["close"]

#     ema_1 = expMovingAverage(close, 14)
#     ema_2 = close.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

#     for [e, r] in zip(ema_1, ema_2):
#         logger.info(f"EMA TEST: [e={e}, r={r}]")


if __name__ == "__main__":
    data = loader.read_resource("btc_sample_sentdex.csv")
    copy = data.copy()

    expected = StochRSIIndicator(copy["close"])
    expected_series = expected.stochrsi_d()

    rolling = StochasticRSI(copy.iloc[:80])
    rolling_series = rolling.d()

    for i, series in copy.iloc[80:].iterrows():
        rolling.update(series)

    for i, [e, r] in enumerate(zip(expected_series, rolling_series)):
        if i < 80:
            logger.info(f"MFI: [i={i}, e={e}, r={r}]")
        else:
            logger.info(f"MFI: [i={i}, e={e}, r_updated={r}]")

# -- Speed Tests --

# if __name__ == "__main__":
#     data = loader.read_resource()
#     copy = data.copy()

#     start = time()
#     iterations = range(10)
#     logger.info("Started.")

#     for iter in iterations:
#         rolling = IchimokuCloud(copy.iloc[:100])
#         logger.info("Updating.")
#         for i, series in copy.iloc[100:].iterrows():
#             rolling.update(series)

#     duration = time() - start
#     logger.info(f"Finished: [duration={duration}, avg_dur={duration/len(iterations)}]")
