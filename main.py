from time import time

import numpy as np
from numba import njit
from numba.types import f8
from rolling_ta.extras.numba import _empty, _mean, _shift


if __name__ == "__main__":
    test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    next_val = 10
    print(test)
    test[:-1] = test[1:]
    test[-1] = next_val
    print(test)

    test2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    next_val = 10
    print(test2)
    test2 = _shift(test2)
    test2[-1] = next_val
    print(test2)
