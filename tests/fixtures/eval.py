from logging import Logger
from typing import Callable
import numpy as np
import pandas as pd
import pytest


Eval = Callable[[np.ndarray[np.float64], np.ndarray[np.float64]], None]


@pytest.fixture(name="evaluate")
def evaluate(log: Logger):
    def e(expected: np.ndarray[np.float64], rolling: np.ndarray[np.float64]):
        if len(expected) != len(rolling):
            pytest.fail(
                f"Length equivalency: [expected={len(expected)}, rolling={len(rolling)}]"
            )
            raise Exception("STOP")

        for i, [e, r] in enumerate(zip(expected, rolling)):
            if not np.isclose(e, r, atol=1e-6):
                log.error(f"Equals: [index={i}, expected={e}, rolling={r}]")
                log.error(f"Equals: [expected=\n{expected}\n, rolling=\n{rolling}\n]")
                pytest.fail(f"Equals: [index={i}, expected={e}, rolling={r}]")
                raise Exception("STOP")

    return e
