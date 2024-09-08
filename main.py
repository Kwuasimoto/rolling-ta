# DO NOT EXPOSE TO PACKAGE.

from rolling_ta.momentum import RSI
from rolling_ta.logging import logger
from ta.momentum import RSIIndicator
import os
import pandas as pd

if __name__ == "__main__":
    data_path = os.path.dirname(__file__)
    file_name = os.path.join(data_path, "tests", "data", "btc_ohlcv.csv")
    data = pd.read_csv(file_name)

    logger.info(f"\n{data[:10]}")

    expected = RSIIndicator(data["close"])
    expected_rsi = expected.rsi()
    logger.info(f"{expected_rsi}")

    rolling = RSI(data)
    rolling_rsi = rolling.data()
    logger.info(f"{rolling_rsi}")
