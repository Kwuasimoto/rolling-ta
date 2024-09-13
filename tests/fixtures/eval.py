from logging import Logger
from typing import Callable
import pandas as pd
import pytest


Eval = Callable[[pd.Series, pd.Series, str], None]


@pytest.fixture(name="evaluate")
def evaluate(log: Logger):
    def e(expected: pd.Series, rolling: pd.Series, error_loc: str):
        for i, [e, r] in enumerate(zip(expected, rolling)):
            if e != r:
                log.error(f"{error_loc}: Failed [index={i}, expected={e}, rolling={r}]")
                pytest.fail(
                    f"{error_loc}: Failed [index={i}, expected={e}, rolling={r}]"
                )

    return e
