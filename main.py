import numpy as np

from rolling_ta.extras.numba import _mfi
from rolling_ta.data import CSVLoader, XLSLoader
from rolling_ta.logging import logger
from ta.volume import MFIIndicator

from rolling_ta.trend.dmi import NumbaDMI


if __name__ == "__main__":
    loader = XLSLoader()
    adx_df = loader.read_resource(
        "cs-adx.xlsx",
        columns=[
            "date",
            "high",
            "low",
            "close",
            "tr",
            "+dm",
            "-dm",
            "tr14",
            "+dm14",
            "-dm14",
            "+di14",
            "-di14",
            "dx",
            "adx",
        ],
    )

    dmi = NumbaDMI(adx_df.iloc[:21])

    for i, series in adx_df.iloc[21:].iterrows():
        dmi.update(series)

    expected_pdmi = adx_df["+di14"].to_numpy(np.float64).round(3)
    rolling_pdmi = dmi.pdmi().round(3)

    logger.info(expected_pdmi)
    logger.info(rolling_pdmi)
