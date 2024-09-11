# DO NOT EXPOSE TO PACKAGE.

import importlib.resources as pkg

import pandas as pd

from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, SMAIndicator, MACD

from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator

from rolling_ta.data import CSVLoader
from rolling_ta.logging import logger

from rolling_ta.volatility import BollingerBands as BB
from rolling_ta.trend import EMA, SMA

from rolling_ta.momentum import RSI
from rolling_ta.volume import MFI

from time import time

# -- Data Loading Tests --

loader = CSVLoader()

# if __name__ == "__main__":
#     values = loader.read_resource()
#     logger.debug(f"\n{values}\n")


# -- Comparison Tests --

if __name__ == "__main__":
    data = loader.read_resource()
    copy = data[:85].copy()

    expected = MFIIndicator(copy["high"], copy["low"], copy["close"], copy["volume"])
    expected_series = expected.money_flow_index()

    rolling = MFI(copy.iloc[:80])
    rolling_series = rolling.mfi()

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
#         rolling = MFI(copy[:28])
#         logger.info("Updating.")
#         for i, series in copy[28:].iterrows():
#             rolling.update(series)

#     duration = time() - start
#     logger.info(f"Finished: [duration={duration}, avg_dur={duration/len(iterations)}]")
