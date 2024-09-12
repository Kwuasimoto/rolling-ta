import importlib.resources as pkg

import pandas as pd

from rolling_ta.data import DataLoader
from rolling_ta.logging import logger


class CSVLoader(DataLoader):

    def read_resource(self, file_name: str = "btc_ohlcv.csv"):
        logger.debug(f"CSVLoader: Loading from resources/{file_name}")
        resources = pkg.files("resources")
        return pd.DataFrame(
            data=pd.read_csv(resources / file_name).values,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

    def read_file(self, path: str):
        raise NotImplementedError("Not implemented yet.")
