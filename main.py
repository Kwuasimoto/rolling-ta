from time import time

import numpy as np
from numba import njit
from numba.types import f8
from rolling_ta.data.csv_loader import CSVLoader
from rolling_ta.extras.numba import _empty, _mean, _shift
from rolling_ta.omni.ichimoku_cloud import NumbaIchimokuCloud


if __name__ == "__main__":
    loader = CSVLoader()
    btc = loader.read_resource()

    slice_a = btc.iloc[:120]
    slice_b = btc.iloc[120:140]

    cloud = NumbaIchimokuCloud(slice_a)

    for _, series in slice_b.iterrows():
        cloud.update(series)

    print(cloud.senkou_b())
