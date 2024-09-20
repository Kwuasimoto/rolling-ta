import numpy as np

from rolling_ta.data import CSVLoader, XLSXLoader, XLSXWriter
from rolling_ta.momentum import NumbaStochasticRSI, NumbaRSI
from ta.momentum import StochRSIIndicator, RSIIndicator

if __name__ == "__main__":
    loader = CSVLoader()
    btc = loader.read_resource()

    writer = XLSXWriter("btc-lrf.xlsx")

    ts = btc["timestamp"].iloc[:200].to_numpy(np.float64)
    high = btc["high"].iloc[:200].to_numpy(np.float64)
    low = btc["low"].iloc[:200].to_numpy(np.float64)
    close = btc["close"].iloc[:200].to_numpy(np.float64)

    writer.write(ts, 1)
    writer.write(high, 2)
    writer.write(low, 3)
    writer.write(close, 4)
