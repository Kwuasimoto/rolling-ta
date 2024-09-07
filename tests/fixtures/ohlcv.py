from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def ohlcv(request: pytest.FixtureRequest, pytestconfig: pytest.Config):
    ohlcv_file = request.config.getoption("--ohlcv")
    prj_root = Path(pytestconfig.rootpath)
    data_path = prj_root / "tests" / "data" / ohlcv_file
    if not data_path.exists():
        pytest.fail(f"OHLCV File not found: {data_path}")
    return pd.read_csv(data_path)
