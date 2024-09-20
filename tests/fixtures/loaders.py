import pytest
from rolling_ta.data import XLSXLoader, CSVLoader


@pytest.fixture(name="xls_loader")
def xls_loader():
    return XLSXLoader()


@pytest.fixture(name="csv_loader")
def csv_loader():
    return CSVLoader()
