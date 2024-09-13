import logging

import pytest

from tests.logging import DataFrameFormatter


@pytest.fixture(name="log")
def log():
    formatter = DataFrameFormatter("%(asctime)s %(levelname)-8s %(message)s", n_rows=4)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
