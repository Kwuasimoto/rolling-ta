# DO NOT EXPOSE TO PACKAGE.

from src.rolling_ta.momentum import RSI
import os
import pandas as pd

if __name__ == "__main__":
    print("yeet")
    data_path = os.path.dirname(__file__)
    file_name = os.path.join(data_path, "tests", "data", "btc_ohlcv.csv")
    data = pd.read_csv(file_name)

    data.info()
