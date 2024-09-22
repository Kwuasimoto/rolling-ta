import numpy as np

from rolling_ta.data import CSVLoader, XLSXLoader, XLSXWriter
from rolling_ta.extras.numba import (
    _linear_regression,
    _linear_regression_forecast,
    _linear_regression_r2,
    _typical_price,
)
from rolling_ta.momentum import StochasticRSI, RSI
from ta.momentum import StochRSIIndicator, RSIIndicator
from rolling_ta.logging import logger
from rolling_ta.volatility.tr import TR
from rolling_ta.volume.mfi import MFI

if __name__ == "__main__":
    # logger.info(not np.isclose(1.000999, 1.000119, atol=1e-6))
    loader = CSVLoader()
    btc = loader.read_resource()
    prices = np.empty(btc["close"].size)
    _typical_price(
        btc["high"].to_numpy(dtype=np.float64),
        btc["low"].to_numpy(dtype=np.float64),
        btc["close"].to_numpy(dtype=np.float64),
        prices,
    )

    intercepts = np.zeros(btc["close"].size, dtype=np.float64)
    slopes = np.zeros(btc["close"].size, dtype=np.float64)

    x_range = np.arange(14)
    x = np.sum(x_range)
    xx = np.sum(x_range * x_range)

    y = np.sum(prices[: x_range.size])
    xy = np.sum(x_range * prices[: x_range.size])

    _linear_regression(prices[:200], slopes[:200], intercepts[:200])
    logger.info(slopes[13:40])

    forecast = 14
    forecasts = np.zeros(slopes[:200].size + forecast)
    _linear_regression_forecast(slopes[:200], intercepts[:200], forecasts)

    r2 = np.zeros(prices[:200].size)
    _linear_regression_r2(prices[:200], slopes[:200], intercepts[:200], r2)

    logger.info(r2)
