import pandas as pd
from logging import getLogger, StreamHandler, INFO
from tests import DataFrameFormatter

# Setup logger to handle pandas dataframes
formatter = DataFrameFormatter("%(asctime)s %(levelname)-8s %(message)s", n_rows=4)
logger = getLogger(__name__)
logger.setLevel(INFO)

ch = StreamHandler()
ch.setFormatter(formatter)

logger.addHandler(ch)


def load_ohlcv(ohlcv: pd.DataFrame):
    logger.info("- OHLCV Dataframe Information -")
    logger.info(f"columns: {ohlcv.columns}")
    logger.info(f"size: {len(ohlcv)}")
    assert ohlcv is not None
