import numpy as np

from rolling_ta.extras.numba import _empty, _shift
from rolling_ta.data import CSVLoader, XLSLoader
from rolling_ta.logging import logger
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import SMAIndicator

from rolling_ta.trend.dmi import NumbaDMI
from rolling_ta.volume import NumbaOBV
from rolling_ta.numba_indicator import NumbaIndicator

from time import time

from numba import typed


if __name__ == "__main__":
    d = typed.Dict()
    d["close"] = typed.List([1.0, 2.0])
    n_indicator = NumbaIndicator(d, True)
    n_indicator.drop_data()
