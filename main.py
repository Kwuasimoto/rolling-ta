# DO NOT EXPOSE TO PACKAGE.

import importlib.resources as pkg

import numpy as np
import pandas as pd

from ta.momentum import StochRSIIndicator
from ta.volume import OnBalanceVolumeIndicator

from rolling_ta.data import CSVLoader, XLSLoader
from rolling_ta.logging import logger

from rolling_ta.volatility import BollingerBands as BB, AverageTrueRange as ATR
from rolling_ta.trend import EMA, SMA, ADX

from rolling_ta.momentum import RSI, StochasticRSI
from rolling_ta.volume import MFI, OBV

from rolling_ta.omni import IchimokuCloud

from time import time

# -- Data Loading Tests --

csv_loader = CSVLoader()
xls_loader = XLSLoader()

# if __name__ == "__main__":
#     values = loader.read_resource()
#     logger.debug(f"\n{values}\n")


# -- Building Zone --

# if __name__ == "__main__":
#     data = loader.read_resource()
#     copy = data[:28].copy()

#     expected = ATR(copy)

# -- Comparison Tests --


# def expMovingAverage(values, window):
#     weights = np.exp(np.linspace(-1.0, 0.0, window))
#     weights /= weights.sum()
#     a = np.convolve(values, weights, mode="full")[: len(values)]
#     a[:window] = a[window]
#     return a


if __name__ == "__main__":
    data = xls_loader.read_resource(
        "cs-obv.xlsx",
        columns=["date", "close", "up-down", "volume", "pos-neg", "obv"],
    ).copy()

    data.info()
    logger.info(f"MAIN: [columns=\n{data.columns}\n]")
    logger.info(f"MAIN: [df_values=\n{data.values}\n]")

    rolling = OBV(data)
    rolling_series = rolling.obv()

    for [i, series] in data.iterrows():
        e = series["obv"]
        r = rolling_series.iloc[i]
        logger.info(f"OBV: [i={i}, e={e}, r={r}]")

    # for i, series in copy.iloc[80:].iterrows():
    #     rolling.update(series)

    # for i, [e, r] in enumerate(zip(expected_series, rolling_series)):
    #     if i < 80:
    #         logger.info(f"MFI: [i={i}, e={e}, r={r}]")
    #     else:
    #         logger.info(f"MFI: [i={i}, e={e}, r_updated={r}]")

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
