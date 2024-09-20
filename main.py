import numpy as np

from rolling_ta.data import CSVLoader, XLSXLoader, XLSXWriter
from rolling_ta.momentum import NumbaStochasticRSI, NumbaRSI
from ta.momentum import StochRSIIndicator, RSIIndicator
from rolling_ta.logging import logger
from rolling_ta.volatility.tr import NumbaTrueRange

if __name__ == "__main__":
    # logger.info(not np.isclose(1.000999, 1.000119, atol=1e-6))
    # loader = CSVLoader()
    # btc = loader.read_resource()

    loader = XLSXLoader()
    atr_df = loader.read_resource(
        "btc-atr.xlsx",
        columns=[
            "ts",
            "high",
            "low",
            "close",
            "tr",
            "atr",
        ],
    ).copy()

    tr = NumbaTrueRange(atr_df.iloc[:40])

    for i, series in atr_df.iloc[40:].iterrows():
        tr.update(series)

    # writer = XLSXWriter("btc-sma.xlsx")

    # ts = btc["timestamp"].iloc[:200].to_numpy(np.float64)
    # close = btc["close"].iloc[:200].to_numpy(np.float64)

    # writer.write(ts, 1)
    # writer.write(close, 2)
