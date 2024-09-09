# Setup logger to handle pandas dataframes
import logging

import pandas as pd

from tests.logging import DataFrameFormatter

formatter = DataFrameFormatter("%(asctime)s %(levelname)-8s %(message)s", n_rows=4)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger.addHandler(ch)


def test_load_ohlcv(ohlcv: pd.DataFrame):
    logger.info("- OHLCV Dataframe Information -")
    logger.info(f"columns: {ohlcv.columns}")
    logger.info(f"size: {len(ohlcv)}")
    assert ohlcv is not None
