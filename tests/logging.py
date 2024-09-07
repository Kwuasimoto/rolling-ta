import pytest
import logging
import pandas as pd


# https://stackoverflow.com/questions/77747229/log-a-dataframe-using-logging-and-pandas
class DataFrameFormatter(logging.Formatter):
    def __init__(self, fmt: str, n_rows: int = 4) -> None:
        self.n_rows = n_rows
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, pd.DataFrame):
            s = ""
            if hasattr(record, "n_rows"):
                self.n_rows = record.n_rows
            lines = record.msg.head(self.n_rows).to_string().splitlines()
            if hasattr(record, "header"):
                record.msg = record.header.strip()
                s += super().format(record) + "\n"
            for line in lines:
                record.msg = line
                s += super().format(record) + "\n"
            return s.strip()
        return super().format(record)


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    handler = logging.FileHandler("test.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
