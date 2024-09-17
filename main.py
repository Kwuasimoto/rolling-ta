import numpy as np

from rolling_ta.extras.numba import _mfi
from rolling_ta.data import CSVLoader
from rolling_ta.logging import logger
from ta.volume import MFIIndicator


if __name__ == "__main__":
    for i in range(0):
        print("Shouldn't run")
    # loader = CSVLoader()
    # data = loader.read_resource().copy().iloc[:20]

    # high = data["high"]
    # low = data["low"]
    # close = data["close"]
    # volume = data["volume"]

    # result, _, __, ___ = _mfi(
    #     high.to_numpy(np.float64),
    #     low.to_numpy(np.float64),
    #     close.to_numpy(np.float64),
    #     volume.to_numpy(np.float64),
    # )
    # expected = MFIIndicator(high, low, close, volume).money_flow_index()

    # logger.info(f"MFI [\n{result}\n]")
    # logger.info(f"MFI [\n{expected}\n]")
