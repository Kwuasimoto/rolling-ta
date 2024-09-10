# DO NOT EXPOSE TO PACKAGE.

import importlib.resources as pkg

import pandas as pd

from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, SMAIndicator

from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator

from rolling_ta.data import CSVLoader
from rolling_ta.logging import logger

from rolling_ta.volatility import BollingerBands as BB
from rolling_ta.trend import EMA, SMA

from rolling_ta.momentum import RSI
from rolling_ta.volume import MFI

# FN Overhead = 6-7 seconds for 100m calls
# def empty_fn():
#     pass

# logger.debug("Start")
# for i in range(0, 100000000):
#     empty_fn()
# logger.debug("Finished")

loader = CSVLoader()

# if __name__ == "__main__":
#     values = loader.read_resource()
#     logger.debug(f"\n{values}\n")

# if __name__ == "__main__":
#     resources = pkg.files("resources")
#     csv_file = "btc_ohlcv.csv"
#     data = pd.read_csv(resources / csv_file)

#     copy = data.loc[:99].copy()

#     expected = None

# if __name__ == "__main__":
#     resources = pkg.files("resources")
#     csv_file = "btc_ohlcv.csv"
#     data = pd.read_csv(resources / csv_file)

#     copy = data.loc[:99].copy()
#     logger.debug(f"BollingerBands Input: [data=\n{copy[:26]}\n]")

#     expected = BollingerBands(copy["close"])
#     expected_bb = expected.bollinger_mavg()

#     rolling = BB(copy)
#     rolling_bb = rolling.data()

#     for [e_sma, r_sma, e_uband, r_uband, e_lband, r_lband] in zip(
#         expected.bollinger_mavg(),
#         rolling_bb["sma_20"],
#         expected.bollinger_hband(),
#         rolling_bb["bb_uband_20"],
#         expected.bollinger_lband(),
#         rolling_bb["bb_lband_20"],
#     ):
#         logger.debug(
#             f"Test: [e_sma={round(e_sma, 2)}, r_sma={round(r_sma, 2)}, e_uband={e_uband}, r_uband={r_uband}, e_lband={e_lband}, r_uband={r_lband}]"
#         )

# if __name__ == "__main__":
#     resources = pkg.files("resources")
#     csv_file = "btc_ohlcv.csv"
#     data = pd.read_csv(resources / csv_file)

#     copy = data.loc[:99].copy()
#     logger.debug(f"SMA Input: [data=\n{copy[:26]}\n]")

#     expected = EMAIndicator(copy["close"])
#     expected_ema = expected.ema_indicator()

#     rolling = EMA(copy)
#     rolling_ema = rolling.data()

#     for i, series in data[99:109].iterrows():
#         rolling.update(series)

#     for [e, r] in zip(expected_ema, rolling_ema["ema_14"]):
#         logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")


# if __name__ == "__main__":
#     data = loader.read_resource()
#     copy = data.loc[:109].copy()

#     expected = SMAIndicator(copy["close"], 12)
#     expected_sma = expected.sma_indicator()

#     rolling = SMA(copy[:99], init=True)
#     rolling_sma = rolling.sma()

#     for i, series in data[99:109].iterrows():
#         rolling.update(series)

#     for [e, r] in zip(expected_sma, rolling_sma):
#         logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")

# # -- RSI (move to tests eventually) --

if __name__ == "__main__":
    data = loader.read_resource()
    copy = data.loc[:109].copy()
    # logger.debug(f"RSI Input: [data=\n{copy[:26]}\n]")

    expected = RSIIndicator(copy["close"])
    expected_rsi = expected.rsi()

    rolling = RSI(copy[:99])
    rolling_rsi = rolling.rsi()

    # print(rolling_rsi)

    for i, series in copy[99:109].iterrows():
        rolling.update(series)

    for i, [e, r] in enumerate(zip(expected_rsi, rolling_rsi)):
        if i <= 99:
            logger.debug(f"Test: [i={i}, e={round(e, 2)}, r={round(r, 2)}]")
        else:
            logger.debug(f"Test: [i={i}, e={round(e, 2)}, r_updated={round(r, 2)}]")
