import importlib.resources as pkg

import pandas as pd

from rolling_ta.data import DataLoader
from rolling_ta.logging import logger


class CSVLoader(DataLoader):

    def read_resource(
        self,
        file_name: str = "btc_ohlcv.csv",
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    ):
        logger.debug(f"CSVLoader: Loading from resources/{file_name}")
        resources = pkg.files("resources")
        df = pd.read_csv(resources / file_name)
        return pd.DataFrame(data=df.values, columns=columns)

    def read_file(self, path: str):
        raise NotImplementedError("Not implemented yet.")
