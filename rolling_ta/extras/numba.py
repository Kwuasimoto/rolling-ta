import numba as nb
import numpy as np

from numba.types import f8, i4


@nb.njit()
def _sma(
    data: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    sma = np.zeros_like(data, dtype=np.float64)
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    sma[period - 1] = current_sum / period
    for i in range(period, data.shape[0]):
        current_sum += data[i] - data[i - period]
        sma[i] = current_sum / period

    return sma, current_sum


@nb.njit
def _sma_update(
    window_sum: f8,
    close: f8,
    close_f: f8,
    period: i4 = 14,
) -> tuple[f8, f8]:
    current_sum = (window_sum - close_f) + close
    return current_sum / period, current_sum


@nb.njit
def _ema(
    data: np.ndarray[f8],
    weight: f8,
    period: i4 = 14,
) -> np.ndarray[f8]:
    ema = np.zeros_like(data, dtype=np.float64)
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    ema_prev = current_sum / period
    ema[period - 1] = ema_prev
    for i in range(period, data.shape[0]):
        ema_latest = ((data[i] - ema_prev) * weight) + ema_prev
        ema[i] = ema_latest
        ema_prev = ema_latest
    return ema


@nb.njit
def _ema_update(
    close: f8,
    weight: f8,
    ema_latest: f8,
) -> np.ndarray[f8]:
    return ((close - ema_latest) * weight) + ema_latest


@nb.njit
def _obv(
    close: np.ndarray[f8],
    volume: np.ndarray[f8],
) -> np.ndarray[f8]:
    obv = np.zeros_like(close)
    close_diff = close[1:] - close[:-1]
    obv_change = np.where(
        close_diff > 0, volume[1:], np.where(close_diff < 0, -volume[1:], 0)
    )
    obv[1:] = np.cumsum(obv_change)
    return obv


@nb.njit
def _obv_update(
    close: f8,
    volume: f8,
    close_p: f8,
    obv_latest: f8,
) -> f8:
    if close > close_p:
        obv_latest += volume
    elif close < close_p:
        obv_latest -= volume
    return obv_latest


@nb.njit(parallel=True)
def _tr(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    close: np.ndarray[f8],
) -> np.ndarray[f8]:
    n = close.size
    close_p = np.empty(n, dtype=np.float64)
    close_p[1:] = close[:-1]
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in nb.prange(1, n):
        hl = high[i] - low[i]
        hc_p = np.abs(high[i] - close_p[i])
        c_pl = close_p[i] - low[i]

        tr[i] = max(hl, hc_p, c_pl)

    return tr


@nb.njit
def _tr_update(
    high: f8,
    low: f8,
    close_p: f8,
) -> f8:
    return max(high - low, np.abs(high - close_p), close_p - low)


@nb.njit
def _atr(
    tr: np.ndarray[f8],
    period: i4 = 14,
    n_1: i4 = 13,
) -> np.ndarray[f8]:
    atr = np.zeros_like(tr)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, tr.size):
        atr[i] = ((atr[i - 1] * n_1) + tr[i]) / period
    return atr


@nb.njit
def _atr_update(
    atr_latest: f8,
    tr_current: f8,
    period: i4 = 14,
    n_1=13,
) -> f8:
    return ((atr_latest * n_1) + tr_current) / period
