import numpy as np

from rolling_ta.logging import logger

from rolling_ta.data import CSVLoader, XLSXLoader, XLSXWriter

from rolling_ta.trend import LinearRegressionR2, lr
from tests.fixtures.data_sheets import lr_df

from ta.volume import VolumeWeightedAveragePrice


def write_xlsx_file():
    loader = CSVLoader()
    btc = loader.read_resource()

    btc_sliced = btc.iloc[:4000].reset_index(drop=True)

    writer = XLSXWriter("btc-bop.xlsx")

    for i, series in enumerate(btc_sliced):
        writer.write(btc_sliced[series].to_numpy(dtype=np.float64), col=i + 1)
        writer.save()


if __name__ == "__main__":
    lr_vwap_df = lr_df(XLSXLoader())
    lr2 = LinearRegressionR2(lr_vwap_df)
    # write_xlsx_file()
    ...
