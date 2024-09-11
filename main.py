# DO NOT EXPOSE TO PACKAGE.

import importlib.resources as pkg

import pandas as pd

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, SMAIndicator, MACD

from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator

from rolling_ta.data import CSVLoader
from rolling_ta.logging import logger

from rolling_ta.volatility import BollingerBands as BB, AverageTrueRange as ATR
from rolling_ta.trend import EMA, SMA

from rolling_ta.momentum import RSI
from rolling_ta.volume import MFI

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

if __name__ == "__main__":
    data = loader.read_resource()
    copy = data[:100].copy()

    expected = AverageTrueRange(copy["high"], copy["low"], copy["close"])
    expected_series = expected.average_true_range()

    rolling = ATR(copy[:80])
    rolling_series = rolling.atr()

    for i, series in copy.iloc[80:].iterrows():
        rolling.update(series)

    for i, [e, r] in enumerate(zip(expected_series, rolling_series)):
        if i < 80:
            logger.info(f"MFI: [i={i}, e={e}, r={r}]")
        else:
            logger.info(f"MFI: [i={i}, e={e}, r_updated={r}]")

# -- Speed Tests --

if __name__ == "__main__":
    data = loader.read_resource()
    copy = data.copy()

    start = time()
    iterations = range(1)
    logger.info("Started.")

    for iter in iterations:
        rolling = ATR(copy)
        # logger.info("Updating.")
        # for i, series in copy[28:].iterrows():
        #     rolling.update(series)

    duration = time() - start
    logger.info(f"Finished: [duration={duration}, avg_dur={duration/len(iterations)}]")
