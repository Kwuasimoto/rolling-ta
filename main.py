# DO NOT EXPOSE TO PACKAGE.

import importlib.resources as pkg

import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator

from rolling_ta.logging import logger
from rolling_ta.volatility import BollingerBands as BB
from rolling_ta.trend import EMA, SMA
from rolling_ta.momentum import RSI


# FN Overhead = 6-7 seconds for 100m calls
# def empty_fn():
#     pass

# logger.debug("Start")
# for i in range(0, 100000000):
#     empty_fn()
# logger.debug("Finished")

if __name__ == "__main__":
    resources = pkg.files("resources")
    csv_file = "btc_ohlcv.csv"
    data = pd.read_csv(resources / csv_file)

    copy = data.loc[:99].copy()
    logger.debug(f"BollingerBands Input: [data=\n{copy[:26]}\n]")

    # expected = BollingerBands(copy["close"])
    # expected_ema = expected.bollinger_mavg()

    rolling = BB(copy)
    rolling_bb = rolling.data()

    # for [e, r] in zip(expected_ema, rolling_ema):
    #     logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")

if __name__ == "__main__":
    resources = pkg.files("resources")
    csv_file = "btc_ohlcv.csv"
    data = pd.read_csv(resources / csv_file)

    copy = data.loc[:99].copy()
    logger.debug(f"SMA Input: [data=\n{copy[:26]}\n]")

    expected = EMAIndicator(copy["close"])
    expected_ema = expected.ema_indicator()

    rolling = EMA(copy)
    rolling_ema = rolling.data()

    for [e, r] in zip(expected_ema, rolling_ema["ema_14"]):
        logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")


if __name__ == "__main__":
    resources = pkg.files("resources")
    csv_file = "btc_ohlcv.csv"
    data = pd.read_csv(resources / csv_file)

    copy = data.loc[:99].copy()
    logger.debug(f"SMA Input: [data=\n{copy[:26]}\n]")

    expected = SMAIndicator(copy["close"], 12)
    expected_sma = expected.sma_indicator()

    rolling = SMA(copy, init=True)
    rolling_sma = rolling.data()

    for [e, r] in zip(expected_sma, rolling_sma["sma_12"]):
        logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")

# -- RSI (move to tests eventually) --

if __name__ == "__main__":
    resources = pkg.files("resources")
    csv_file = "btc_ohlcv.csv"
    data = pd.read_csv(resources / csv_file)

    copy = data.loc[:99].copy()
    logger.debug(f"RSI Input: [data=\n{copy[:26]}\n]")

    expected = RSIIndicator(copy["close"])
    expected_rsi = expected.rsi()

    rolling = RSI(copy)
    rolling_rsi = rolling.data()

    print(rolling_rsi)

    for [e, r] in zip(expected_rsi, rolling_rsi["rsi_14"]):
        logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")
