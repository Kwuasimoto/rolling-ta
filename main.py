import numpy as np

from rolling_ta.data import CSVLoader, XLSXLoader, XLSXWriter
from rolling_ta.momentum import StochasticRSI, RSI
from ta.momentum import StochRSIIndicator, RSIIndicator
from rolling_ta.logging import logger
from rolling_ta.volatility.tr import TrueRange
from rolling_ta.volume.mfi import MFI

if __name__ == "__main__":
    # logger.info(not np.isclose(1.000999, 1.000119, atol=1e-6))
    # loader = CSVLoader()
    # btc = loader.read_resource()

    loader = XLSXLoader()
    mfi_df = loader.read_resource(
        "btc-mfi.xlsx",
        columns=[
            "ts",
            "high",
            "low",
            "close",
            "typical",
            "volume",
            "rmf",
            "pmf",
            "nmf",
            "pmf_sum_14",
            "nmf_sum_14",
            "mfi",
        ],
    ).copy()
    mfi_expected = mfi_df["mfi"].to_numpy(dtype=np.float64)
    mfi = MFI(mfi_df.iloc[:20])

    for [e, [i, series]] in zip(mfi_expected[20:], mfi_df.iloc[20:].iterrows()):
        mfi.update(i, series)
        r = mfi._mfi[i]

        if not np.isclose(e, r, atol=1e-7):
            logger.info(f"Oof [i={i}, e={e}, r={r}]")

    # slice_b = mfi_df.iloc[40:]

    # for i, series in slice_b.iterrows():
    #     mfi.update(series)

    # writer = XLSXWriter("btc-sma.xlsx")

    # ts = btc["timestamp"].iloc[:200].to_numpy(np.float64)
    # close = btc["close"].iloc[:200].to_numpy(np.float64)

    # writer.write(ts, 1)
    # writer.write(close, 2)
