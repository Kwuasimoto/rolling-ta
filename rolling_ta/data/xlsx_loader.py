import pandas as pd
from rolling_ta.data import DataLoader

from rolling_ta.logging import logger

import importlib.resources as pkg


class XLSXLoader(DataLoader):

    def read_resource(
        self,
        file_name: str,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    ):
        logger.debug(f"XLSXLoader: Loading from resources/{file_name}")
        resources = pkg.files("resources")
        df = pd.read_excel(resources / file_name, header=None)
        return pd.DataFrame(data=df.values, columns=columns)

    def read_file(self, path: str):
        return NotImplementedError("Not implemented yet.")
