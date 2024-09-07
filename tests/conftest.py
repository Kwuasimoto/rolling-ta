from pytest import Parser
import logging

pytest_plugins = ["logging", "tests.fixtures.ohlcv"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pytest_addoption(parser: Parser):
    logger.info("Loading options")
    parser.addoption(
        "--ohlcv",
        action="store",
        default="btc_ohlcv.csv",
        help="Path to ohlcv data to perform indicator calculations on. (Should be in format [Timestamp(Seconds), Open, High, Low, Close, Volume]) \nPlease place the file under ./tests/data/[Your file here]",
    )
