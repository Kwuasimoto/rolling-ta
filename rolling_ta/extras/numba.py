from re import S
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
        ema_prev = ((data[i] - ema_prev) * weight) + ema_prev
        ema[i] = ema_prev
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


@nb.njit
def _dm(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
) -> tuple[np.ndarray[f8], np.ndarray[f8], f8, f8]:
    high_p = np.empty(high.shape, dtype=np.float64)
    high_p[1:] = high[:-1]

    low_p = np.empty(low.shape, dtype=np.float64)
    low_p[1:] = low[:-1]

    high[0] = 0.0
    low[0] = 0.0
    high_p[0] = 0.0
    low_p[0] = 0.0

    move_up = high - high_p
    move_down = low_p - low

    move_up_mask = (move_up > 0) & (move_up > move_down)
    move_down_mask = (move_down > 0) & (move_down > move_up)

    pdm = np.zeros(move_up_mask.shape, dtype=np.float64)
    ndm = np.zeros(move_down_mask.shape, dtype=np.float64)

    pdm[move_up_mask] = move_up[move_up_mask]
    ndm[move_down_mask] = move_down[move_down_mask]

    return pdm, ndm, pdm[-1], ndm[-1]


@nb.njit
def _dm_update(
    high: f8,
    low: f8,
    high_p: f8,
    low_p: f8,
) -> tuple[f8, f8]:
    move_up = high - high_p
    move_down = low_p - low

    move_up_bool = (move_up > 0) & (move_up > move_down)
    move_down_bool = (move_down > 0) & (move_down > move_up)

    pdm = move_up if move_up_bool else 0.0
    ndm = move_down if move_down_bool else 0.0

    return pdm, ndm


@nb.njit
def _dm_smoothing(
    dm: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    # According to: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx
    # The initial TrueRange value (ex: high - low) is not a valid True Range, so we start at period + 1
    p_1 = period + 1
    s = np.zeros(dm.shape, dtype=np.float64)
    s[p_1 - 1] = np.sum(dm[1:p_1])
    for i in range(p_1, dm.size):
        s_p = s[i - 1]
        s[i] = s_p - (s_p / period) + dm[i]
    return s, s[-1]


@nb.njit
def _dm_smoothing_update(
    dm: f8,
    dm_p: f8,
    period: i4 = 14,
) -> f8:
    return dm_p - (dm_p / period) + dm


@nb.njit
def _dmi(
    dm: np.ndarray[f8],
    tr: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    dmi = np.zeros(dm.shape, dtype=np.float64)
    dmi[period:] = (dm[period:] / tr[period:]) * 100
    return dmi, dmi[-1]


@nb.njit
def _dmi_update(
    dm: f8,
    tr: f8,
) -> f8:
    return (dm / tr) * 100


@nb.njit
def _dx(
    pdmi: np.ndarray[f8],
    ndmi: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    dx = np.zeros(pdmi.shape, dtype=np.float64)
    dx[period:] = (
        np.abs(pdmi[period:] - ndmi[period:]) / (pdmi[period:] + ndmi[period:])
    ) * 100
    return dx, dx[-1]


@nb.njit
def _dx_update(
    pdmi: f8,
    ndmi: f8,
) -> f8:
    delta = pdmi - ndmi
    if delta == 0:
        return 0
    return (np.abs(delta) / (pdmi + ndmi)) * 100


@nb.njit
def _adx(
    dx: np.ndarray[f8],
    adx_period: i4 = 14,
    dmi_period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    pp = adx_period + dmi_period
    weight = adx_period - 1
    adx = np.zeros(dx.shape, dtype=np.float64)
    adx[pp - 1] = np.mean(dx[adx_period:pp])
    for i in range(pp, adx.size):
        adx[i] = ((adx[i - 1] * weight) + dx[i]) / adx_period
    return adx, adx[-1]


@nb.njit
def _adx_update(
    dx: f8,
    adx_p: f8,
    adx_period: i4 = 14,
) -> f8:
    return ((adx_p * (adx_period - 1)) + dx) / adx_period
