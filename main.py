from time import time

import numpy as np
from numba import njit
from numba.types import f8
from rolling_ta.extras.numba import _empty, _mean


if __name__ == "__main__":
    test = np.random.rand(50)

    result = np.max(test[:14])
