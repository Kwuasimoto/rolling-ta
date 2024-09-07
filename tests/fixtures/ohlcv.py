from pathlib import Path
import logging
import pandas as pd
import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture
def ohlcv(request: pytest.FixtureRequest, pytestconfig: pytest.Config):
    ohlcv_file = request.config.getoption("--ohlcv")
    logger.info(f"--ohlcv = {ohlcv_file}")
    prj_root = Path(pytestconfig.rootpath)
    data_path = prj_root / "tests" / "data" / ohlcv_file
    if not data_path.exists():
        pytest.fail(f"OHLCV File not found: {data_path}")
    return pd.read_csv(data_path)
