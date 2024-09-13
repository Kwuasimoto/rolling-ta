import pytest
from rolling_ta.data import XLSLoader, CSVLoader


@pytest.fixture(name="xls_loader")
def xls_loader():
    return XLSLoader()


@pytest.fixture(name="csv_loader")
def csv_loader():
    return CSVLoader()
