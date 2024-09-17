import os
import numba as nb
import numpy as np

from numba.types import f8, f4, i4, i2, i8, b1

## // HELPER FUNCTIONS \\
## Note: The outer njit function *supposedly* does not need to be supplied parallel=True.


@nb.njit(parallel=True)
def _shift(arr: np.ndarray[f8], shift: i8 = -1) -> np.ndarray[f8]:
    shifted = np.empty(arr.size, dtype=np.float64)
    for i in nb.prange(-shift, arr.size):
        shifted[i] = arr[shift]
    return shifted


@nb.njit(parallel=True, inline="always")
def _prefix_sum(arr: np.ndarray[i4]) -> np.ndarray[f8]:
    prefix_sum = np.empty(arr.shape, dtype=np.int32)
    prefix_sum[0] = arr[0]
    for i in nb.prange(1, arr.size):
        prefix_sum[i] = arr[i] + prefix_sum[i - 1]

    return prefix_sum


@nb.njit(parallel=True, inline="always")
def _mean(arr: np.ndarray[f8]) -> f8:
    n = arr.size
    sum: f8 = 0.0
    for i in nb.prange(n):
        sum += arr[i]
    return sum / n


@nb.njit(parallel=True, inline="always")
def _empty(
    size: i8, fill_zeros: i8 = 0, dtype: np.dtype = np.float64
) -> np.ndarray[f8]:
    arr: np.ndarray[f8] = np.empty(size, dtype=dtype)
    for i in nb.prange(fill_zeros):
        arr[i] = 0
    return arr


@nb.njit(parallel=True, inline="always")
def _sliding_midpoint(
    high: np.ndarray[f8], low: np.ndarray[f8], period: i4
) -> np.ndarray:
    n: i8 = high.size
    arr: np.ndarray[f8] = _empty(n, period, dtype=np.float64)

    for i in nb.prange(period - 1, n):
        max_val: f8 = 0.0
        min_val: f8 = np.inf

        for j in range(i - period + 1, i + 1):
            if high[j] > max_val:
                max_val = high[j]
            if low[j] < min_val:
                min_val = low[j]

        arr[i] = (max_val + min_val) * 0.5

    return arr


## // INDICATOR FUNCTIONS \\


@nb.njit(parallel=True)
def _sma(
    data: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    sma: np.ndarray[f8] = _empty(data.size, period)
    current_sum: f8 = 0.0
    for i in nb.prange(period):
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


@nb.njit(parallel=True)
def _ema(
    data: np.ndarray[f8],
    weight: f8,
    period: i4 = 14,
) -> np.ndarray[f8]:
    ema = _empty(data.size, period)
    current_sum = 0.0
    for i in nb.prange(period):
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


@nb.njit(parallel=True)
def _obv(
    close: np.ndarray[f8],
    volume: np.ndarray[f8],
) -> np.ndarray[f8]:
    n = close.size
    obv = _empty(n, dtype=np.float64)
    obv[0] = 0.0

    for i in nb.prange(1, n):
        curr_close = close[i]
        prev_close = close[i - 1]

        if curr_close > prev_close:
            obv[i] = volume[i]
        elif curr_close < prev_close:
            obv[i] = -volume[i]
        else:
            obv[i] = 0

    return _prefix_sum(obv)


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
def _mfi(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    close: np.ndarray[f8],
    volume: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f4], np.ndarray[f8], np.ndarray[f8], f8]:
    # MFI should be between 0 and 100, which is f2 compatible.
    n = close.size

    typical_price = _empty(n, dtype=np.float64)
    mfp = _empty(n, period, dtype=np.float64)
    mfn = _empty(n, period, dtype=np.float64)
    mfi = _empty(n, period, dtype=np.float32)

    for i in nb.prange(n):
        typical_price[i] = (high[i] + low[i] + close[i]) / 3

    for i in nb.prange(1, n):
        curr_typical = typical_price[i]
        prev_typical = typical_price[i - 1]

        if curr_typical > prev_typical:
            mfp[i] = curr_typical * volume[i]
        elif curr_typical < prev_typical:
            mfn[i] = curr_typical * volume[i]

    n = close.size
    for i in nb.prange(0, n - period + 1):
        i_p = i + period - 1
        mfp_sum = 0.0
        mfn_sum = 0.0

        for j in nb.prange(i, i + period):
            if mfp[j] > 0:
                mfp_sum += mfp[j]
            elif mfn[j] > 0:
                mfn_sum += mfn[j]

        if mfp_sum == 0:
            mfi[i_p] = 0
        elif mfn_sum == 0:
            mfi[i_p] = 100
        else:
            mfi[i_p] = (100 * mfp_sum) / (mfp_sum + mfn_sum)

    return mfi, mfp[-period:], mfn[-period:], typical_price[-1]


@nb.njit
def _mfi_update(high: f8, low: f8, close: f8, prev_typical: f8) -> tuple[f8, f8]:

    typical_price = (high + low + close) / 3

    return 0, 0


@nb.njit(parallel=True)
def _tr(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    close: np.ndarray[f8],
) -> np.ndarray[f8]:
    n = close.size

    close_p = _empty(n, dtype=np.float64)
    tr = _empty(n, dtype=np.float64)

    close_p[1:] = close[:-1]
    tr[0] = high[0] - low[0]

    for i in nb.prange(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close_p[i]), close_p[i] - low[i])

    return tr


@nb.njit
def _tr_update(
    high: f8,
    low: f8,
    close_p: f8,
) -> f8:
    return max(high - low, abs(high - close_p), close_p - low)


@nb.njit(parallel=True)
def _atr(
    tr: np.ndarray[f8],
    period: i4 = 14,
    n_1: i4 = 13,
) -> np.ndarray[f8]:
    atr = _empty(tr.size, period, dtype=np.float64)
    atr[period - 1] = _mean(tr[:period])
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
    high_p = _empty(high.size, dtype=np.float64)
    high_p[1:] = high[:-1]

    low_p = _empty(low.size, dtype=np.float64)
    low_p[1:] = low[:-1]

    high[0] = 0.0
    low[0] = 0.0
    high_p[0] = 0.0
    low_p[0] = 0.0

    move_up = high - high_p
    move_down = low_p - low

    move_up_mask = (move_up > 0) & (move_up > move_down)
    move_down_mask = (move_down > 0) & (move_down > move_up)

    pdm = np.zeros(move_up_mask.size, dtype=np.float64)
    ndm = np.zeros(move_down_mask.size, dtype=np.float64)

    pdm[move_up_mask] = move_up[move_up_mask]
    ndm[move_down_mask] = move_down[move_down_mask]

    return pdm, ndm, high[-1], low[-1]


# OK
@nb.njit
def _dm_update(
    high: f8,
    low: f8,
    high_p: f8,
    low_p: f8,
) -> tuple[f8, f8]:
    move_up = high - high_p
    move_down = low_p - low

    move_up_mask = (move_up > 0) & (move_up > move_down)
    move_down_mask = (move_down > 0) & (move_down > move_up)

    pdm = move_up if move_up_mask else 0.0
    ndm = move_down if move_down_mask else 0.0

    return pdm, ndm


@nb.njit(parallel=True)
def _dm_smoothing(
    x: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    # According to: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx
    # The initial TrueRange value (ex: high - low) is not a valid True Range, so we start at period + 1
    p_1 = period + 1
    s = _empty(x.size, period, dtype=np.float64)
    x_sum: i8 = 0

    for i in nb.prange(1, p_1):
        x_sum += x[i]
    s[period] = x_sum

    for i in range(p_1, x.size):
        s_p = s[i - 1]
        s[i] = s_p - (s_p / period) + x[i]

    return s, s[-1]


@nb.njit
def _dm_smoothing_update(
    x: f8,
    s_p: f8,
    period: i4 = 14,
) -> f8:
    return s_p - (s_p / period) + x


@nb.njit(parallel=True)
def _dmi(
    dm: np.ndarray[f8],
    tr: np.ndarray[f8],
    period: i4 = 14,
) -> np.ndarray[f8]:
    n = dm.size
    dmi = _empty(dm.size, period, dtype=np.float64)

    for i in nb.prange(period, n):
        dmi[i] = (dm[i] / tr[i]) * 100

    return dmi


@nb.njit
def _dmi_update(
    dm: f8,
    tr: f8,
) -> f8:
    return (dm / tr) * 100


@nb.njit(parallel=True)
def _dx(
    pdmi: np.ndarray[f8],
    ndmi: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    n = pdmi.size
    dx = _empty(n, period, dtype=np.float64)

    for i in nb.prange(period, n):
        dx[i] = (abs(pdmi[i] - ndmi[i]) / (pdmi[i] + ndmi[i])) * 100

    return dx, dx[-1]


@nb.njit
def _dx_update(
    pdmi: f8,
    ndmi: f8,
) -> f8:
    delta = pdmi - ndmi
    if delta == 0:
        return 0
    return (abs(pdmi - ndmi) / (pdmi + ndmi)) * 100


@nb.njit(parallel=True)
def _adx(
    dx: np.ndarray[f8],
    adx_period: i4 = 14,
    dmi_period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    pp: i4 = adx_period + dmi_period
    weight: i4 = adx_period - 1

    adx: np.ndarray[f8] = _empty(dx.size, pp, dtype=np.float64)
    adx_0: f8 = 0.0

    for i in nb.prange(adx_period, pp):
        adx_0 += dx[i]

    adx[pp - 1] = adx_0 / adx_period

    for i in range(pp, adx.size):
        adx[i] = ((adx[i - 1] * weight) + dx[i]) / adx_period

    return adx, adx[-1]


@nb.njit
def _adx_update(
    dx: f8,
    adx_p: f8,
    adx_period: i4 = 14,
    n_1: id = 13,
) -> f8:
    return ((adx_p * n_1) + dx) / adx_period


@nb.njit(parallel=True, inline="always")
def _highs_lows(
    high: np.ndarray[f8], low: np.ndarray[f8], period: i4, to_range: i4
) -> tuple[np.ndarray[f8], np.ndarray[f8]]:
    highs = np.empty(high.shape, dtype=np.float64)
    lows = np.empty(low.shape, dtype=np.float64)

    for i in nb.prange(period, to_range):
        max_high = 0
        min_low = np.inf
        for j in range(i - period, i):
            if high[j] > max_high:
                max_high = high[j]
            if low[j] < min_low:
                min_low = low[j]
        highs[i - 1] = max_high
        lows[i - 1] = min_low

    return highs, lows


@nb.njit(parallel=True)
def _ichimoku_cloud(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_period: int = 52,
) -> tuple[
    np.ndarray[f8],
    np.ndarray[f8],
    np.ndarray[f8],
    np.ndarray[f8],
    np.ndarray[f8],
    np.ndarray[f8],
    f8,
    f8,
    f8,
    f8,
]:
    n = high.size
    to_range = n + 1
    a_start = max(tenkan_period, kijun_period) - 1

    tenkan_highs = _empty(n, tenkan_period, dtype=np.float64)
    tenkan_lows = _empty(n, tenkan_period, dtype=np.float64)
    kijun_highs = _empty(n, kijun_period, dtype=np.float64)
    kijun_lows = _empty(n, kijun_period, dtype=np.float64)
    senkou_highs = _empty(n, senkou_period, dtype=np.float64)
    senkou_lows = _empty(n, senkou_period, dtype=np.float64)
    tenkan = _empty(n, tenkan_period, dtype=np.float64)
    kijun = _empty(n, kijun_period, dtype=np.float64)
    senkou_a = _empty(n, senkou_period, dtype=np.float64)
    senkou_b = _empty(n, senkou_period, dtype=np.float64)

    tenkan_highs, tenkan_lows = _highs_lows(high, low, tenkan_period, to_range)
    kijun_highs, kijun_lows = _highs_lows(high, low, kijun_period, to_range)
    senkou_highs, senkou_lows = _highs_lows(high, low, senkou_period, to_range)

    for i in nb.prange(n):
        tenkan[i] = (tenkan_highs[i] + tenkan_lows[i]) * 0.5
        kijun[i] = (kijun_highs[i] + kijun_lows[i]) * 0.5
        senkou_b[i] = (senkou_highs[i] + senkou_lows[i]) * 0.5

    for i in nb.prange(a_start, n):
        senkou_a[i] = (tenkan[i] + kijun[i]) * 0.5

    senkou_a[:a_start] = senkou_a[a_start]
    senkou_b[: senkou_period - 1] = senkou_b[senkou_period - 1]

    greatest_period = max(tenkan_period, kijun_period, senkou_period)

    return (
        tenkan,
        kijun,
        senkou_a,
        senkou_b,
        high[-greatest_period:],
        low[-greatest_period:],
        tenkan[-1],
        kijun[-1],
        senkou_a[-1],
        senkou_b[-1],
    )


@nb.njit
def _ichimoku_cloud_update(
    high: np.ndarray[f8],
    next_high: f8,
    low: np.ndarray[f8],
    next_low: f8,
    tenkan_period: i4 = 9,
    kijun_period: i4 = 26,
    senkou_period: i4 = 52,
) -> tuple[f8, f8, f8, f8, np.ndarray[f8], np.ndarray[f8]]:

    high = _shift(high)
    high[-1] = next_high

    low = _shift(low)
    low[-1] = next_low

    # Compute tenkan, kijun, senkou_a, senkou_b values
    tenkan = (max(high[-tenkan_period:]) + min(low[-tenkan_period:])) * 0.5
    kijun = (max(high[-kijun_period:]) + min(low[-kijun_period:])) * 0.5

    senkou_a = (tenkan + kijun) * 0.5
    senkou_b = (max(high[-senkou_period:]) + min(low[-senkou_period:])) * 0.5

    # Return the updated values and arrays
    return tenkan, kijun, senkou_a, senkou_b, high, low
