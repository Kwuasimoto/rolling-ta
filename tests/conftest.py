import logging
from pathlib import Path

import pandas as pd
import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pytest_plugins = ["logging", "tests.fixtures.ohlcv"]


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--ohlcv",
        action="store",
        default="btc_ohlcv.csv",
        help="Path to ohlcv data to perform indicator calculations on. (Should be in format [Timestamp(Seconds), Open, High, Low, Close, Volume]) \nPlease place the file under ./tests/data/[Your file here]",
    )
