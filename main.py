# DO NOT EXPOSE TO PACKAGE.

from rolling_ta.trend import EMA
from rolling_ta.logging import logger
from ta.trend import EMAIndicator
import os
import pandas as pd

if __name__ == "__main__":
    data_path = os.path.dirname(__file__)
    file_name = os.path.join(data_path, "tests", "data", "btc_ohlcv.csv")
    data = pd.read_csv(file_name)

    copy = data.loc[:99].copy()
    logger.debug(f"SMA Input: [data=\n{copy[:26]}\n]")

    expected = EMAIndicator(copy["close"])
    expected_ema = expected.ema_indicator()

    rolling = EMA(copy)
    rolling_ema = rolling.data()

    for [e, r] in zip(expected_ema, rolling_ema):
        logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")


# if __name__ == "__main__":
#     data_path = os.path.dirname(__file__)
#     file_name = os.path.join(data_path, "tests", "data", "btc_ohlcv.csv")
#     data = pd.read_csv(file_name)

#     copy = data.loc[:99].copy()
#     logger.debug(f"SMA Input: [data=\n{copy[:26]}\n]")

#     expected = SMAIndicator(copy["close"], 12)
#     expected_sma = expected.sma_indicator()

#     rolling = SMA(copy, init=True)
#     rolling_sma = rolling.data()

#     for [e, r] in zip(expected_sma, rolling_sma):
#         logger.debug(f"Test: [e={round(e, 2)}, r={round(r, 2)}]")

## -- RSI (move to tests eventually) --

# if __name__ == "__main__":
#     data_path = os.path.dirname(__file__)
#     file_name = os.path.join(data_path, "tests", "data", "btc_ohlcv.csv")
#     data = pd.read_csv(file_name)

#     logger.info(f"\n{data[:10]}")

#     expected = RSIIndicator(data["close"])
#     expected_rsi = expected.rsi()
#     logger.info(f"{expected_rsi}")

#     rolling = RSI(data)
#     rolling_rsi = rolling.data()
#     logger.info(f"{rolling_rsi}")
