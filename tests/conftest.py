import os
import sys
import pytest

pytest_plugins = ["tests.fixtures"]


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--data-file-name",
        action="store",
        default="btc_ohlcv.csv",
        help="Path to ohlcv data to perform indicator calculations on. (Should be in format [Timestamp(Seconds), Open, High, Low, Close, Volume]) \nPlease place the file under ./tests/data/[Your file here]",
    )
