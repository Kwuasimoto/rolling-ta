import importlib.resources as pkg
import os
from pathlib import Path
from typing import List

import pandas as pd

from rolling_ta.data import DataLoader
from rolling_ta.logging import log


class CSVLoader(DataLoader):

    def read_resource(
        self,
        file_name: str = "btc_ohlcv.csv",
        file_path: List[str] = ["data"],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        index=["timestamp"],
    ):
        log.debug(f"CSVLoader: Loading from resources/{file_name}")
        resources = pkg.files("resources")
        df = pd.read_csv(resources / Path(*file_path) / file_name)
        df.columns = columns

        if not set(index).issubset(columns):
            raise ValueError(
                f" Index column {index} is not a specified column {columns}"
            )

        df.set_index(index, inplace=True)

        return df

    def read_file(self, path: str):
        raise NotImplementedError("Not implemented yet.")
