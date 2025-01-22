import pandas as pd
from rolling_ta.data import DataLoader

from rolling_ta.logging import log

import importlib.resources as pkg


class XLSXLoader(DataLoader):

    def read_resource(
        self,
        file_name: str = "btc_200.xlsx",
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        index=["timestamp"],
    ):
        log.debug(f"XLSXLoader: Loading from resources/{file_name}")
        resources = pkg.files("resources")

        df = pd.read_excel(resources / file_name, header=None)
        df.columns = columns

        if not set(index).issubset(columns):
            raise ValueError(
                f" Index column {index} is not a specified column {columns}"
            )

        df.set_index(index, inplace=True)

        return df

    def read_file(self, path: str):
        return NotImplementedError("Not implemented yet.")
