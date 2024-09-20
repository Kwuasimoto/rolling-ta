import numpy as np

from rolling_ta.data import CSVLoader, XLSXLoader, XLSXWriter
from rolling_ta.momentum import NumbaStochasticRSI, NumbaRSI
from ta.momentum import StochRSIIndicator, RSIIndicator
from rolling_ta.logging import logger

if __name__ == "__main__":
    logger.info(not np.isclose(1.000999, 1.000119, atol=1e-6))
    # loader = CSVLoader()
    # btc = loader.read_resource()

    # writer = XLSXWriter("btc-sma.xlsx")

    # ts = btc["timestamp"].iloc[:200].to_numpy(np.float64)
    # close = btc["close"].iloc[:200].to_numpy(np.float64)

    # writer.write(ts, 1)
    # writer.write(close, 2)
