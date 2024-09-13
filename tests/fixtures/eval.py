from logging import Logger
from typing import Callable
import pandas as pd
import pytest


Eval = Callable[[pd.Series, pd.Series], None]


@pytest.fixture(name="evaluate")
def evaluate(log: Logger):
    def e(expected: pd.Series, rolling: pd.Series):
        for i, [e, r] in enumerate(zip(expected, rolling)):
            if e != r:
                log.error(f"Test failed: [index={i}, expected={e}, rolling={r}]")
                pytest.fail(f"Test failed: [index={i}, expected={e}, rolling={r}]")

    return e
