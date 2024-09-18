from functools import cache
import os
import numba as nb
import numpy as np

from numba.types import f8, f4, i4, i2, i8, b1
import numba.types as ntypes
import numba.typed as ntyped

from rolling_ta.env import NUMBA_DISK_CACHING


## // HELPER FUNCTIONS \\
## Note: The outer njit function *supposedly* does not need to be supplied parallel=True.


@nb.njit(
    parallel=True,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _shift(
    arr: np.ndarray[f8],
    shift: i4 = -1,
    dtype: np.ndarray[f8] = np.float64,
) -> np.ndarray[f8]:
    n = arr.size

    shifted = _empty(n, dtype=dtype)

    if shift == 0:
        raise ValueError("_shift: shift parameter cannot be 0")

    if shift > 0:
        for i in nb.prange(shift, n):
            shifted[i] = arr[i - shift]
        for i in nb.prange(shift):
            shifted[i] = 0
    else:
        n_shift = n + shift

        for i in nb.prange(n_shift):
            shifted[i] = arr[i - shift]
        for i in nb.prange(n_shift, n):
            shifted[i] = 0

    return shifted


@nb.njit(
    parallel=True,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _highs_lows(
    high: np.ndarray[f8], low: np.ndarray[f8], period: i4, to_range: i4
) -> tuple[np.ndarray[f8], np.ndarray[f8]]:
    n = high.shape[0]
    highs = _empty(n, period, dtype=np.float64)  # Initialize all values to zero
    lows = _empty(n, period, dtype=np.float64)  # Initialize lows to infinity

    for i in nb.prange(period, to_range):
        max_high = np.max(high[i - period : i])  # Use vectorized np.max
        min_low = np.min(low[i - period : i])  # Use vectorized np.min
        highs[i - 1] = max_high
        lows[i - 1] = min_low

    return highs, lows


@nb.njit(
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _prefix_sum(arr: np.ndarray[f8]) -> np.ndarray[f8]:
    n = arr.size
    prefix_sum = arr.copy()
    prefix_sum[0] = arr[0]
    for i in range(1, n):
        prefix_sum[i] = arr[i] + prefix_sum[i - 1]
    return prefix_sum


@nb.njit(
    parallel=True,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _mean(arr: np.ndarray[f8]) -> f8:
    n = arr.size
    sum: f8 = 0.0
    for i in nb.prange(n):
        sum += arr[i]
    return sum / n


@nb.njit(
    parallel=True,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _empty(
    size: i4,
    insert_n_zeros: i4 = 0,
    dtype: np.dtype = np.float64,
) -> np.ndarray[f8]:
    arr = np.empty(size, dtype=dtype)
    for i in nb.prange(insert_n_zeros):
        arr[i] = 0
    return arr


@nb.njit(
    parallel=True,
    inline="always",
    cache=NUMBA_DISK_CACHING,
)
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


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _sma(
    data: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], np.ndarray[f8], f8, f8]:
    sma: np.ndarray[f8] = _empty(data.size, period)
    current_sum: f8 = 0.0
    for i in nb.prange(period):
        current_sum += data[i]
    sma[period - 1] = current_sum / period
    for i in range(period, data.size):
        current_sum += data[i] - data[i - period]
        sma[i] = current_sum / period
    return sma, data[-period:], current_sum, sma[-1]


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _sma_update(
    close: f8,
    window_sum: f8,
    window: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8, f8]:
    first = window[0]
    window = _shift(window)
    window[-1] = close
    window_sum = (window_sum - first) + close
    return window_sum / period, window, window_sum


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _ema_update(
    close: f8,
    weight: f8,
    ema_latest: f8,
) -> np.ndarray[f8]:
    return ((close - ema_latest) * weight) + ema_latest


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _rsi(
    close: np.ndarray[f8],
    rsi_container: np.ndarray[f8],
    gains_container: np.ndarray[f8],
    losses_container: np.ndarray[f8],
    period: i4 = 14,
    p_1: i4 = 13,
) -> tuple[np.ndarray[f8], f8, f8, f8]:
    n = close.size

    # Phase 1 (SMA)
    for i in nb.prange(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains_container[i] = delta
        elif delta < 0:
            losses_container[i] = -delta

    avg_gain = _mean(gains_container[1 : period + 1])
    avg_loss = _mean(losses_container[1 : period + 1])

    rsi_container[period] = (100 * avg_gain) / (avg_gain + avg_loss)

    # Phase 2 (EMA)
    for i in range(period + 1, n):
        avg_gain = ((avg_gain * p_1) + gains_container[i]) / period
        avg_loss = ((avg_loss * p_1) + losses_container[i]) / period
        rsi_container[i] = (100 * avg_gain) / (avg_gain + avg_loss)

    return rsi_container, avg_gain, avg_loss, close[-1]


@nb.njit(
    cache=True,
    fastmath=True,
)
def _rsi_update(
    close: f8,
    prev_close: f8,
    avg_gain: f8,
    avg_loss: f8,
    p_1: f8 = 13,
) -> tuple[f8, f8, f8]:
    delta = close - prev_close

    gain = max(delta, 0)
    loss = -min(delta, 0)

    avg_gain = p_1 * (gain - avg_gain) + avg_gain
    avg_loss = p_1 * (loss - avg_loss) + avg_loss

    rsi = (100 * avg_gain) / (avg_gain + avg_loss)

    return rsi, avg_gain, avg_loss


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _stoch_rsi(
    rsi: np.ndarray[f8],
    window: np.ndarray[f8],
    period: i4,
) -> np.ndarray[f8]:
    n = rsi.size
    window = rsi[:period]
    stoch_rsi = np.zeros(n, dtype=np.float64)

    for i in nb.prange(period, n):
        pass


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _obv(
    close: np.ndarray[f8],
    volume: np.ndarray[f8],
    obv_container: np.ndarray[f8],
) -> np.ndarray[f8]:
    n = close.size

    for i in nb.prange(1, n):
        curr_close = close[i]
        prev_close = close[i - 1]

        if curr_close > prev_close:
            obv_container[i] = volume[i]
        elif curr_close < prev_close:
            obv_container[i] = -volume[i]
        else:
            obv_container[i] = 0

    obv = _prefix_sum(obv_container)

    return obv, obv[-1], close[-1]


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _mfi_update(high: f8, low: f8, close: f8, prev_typical: f8) -> tuple[f8, f8]:

    typical_price = (high + low + close) / 3

    return 0, 0


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _tr_update(
    high: f8,
    low: f8,
    close_p: f8,
) -> f8:
    return max(high - low, abs(high - close_p), close_p - low)


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _atr(
    tr: np.ndarray[f8],
    period: i4 = 14,
    n_1: i4 = 13,
) -> tuple[np.ndarray[f8], f8]:
    atr = _empty(tr.size, period, dtype=np.float64)
    atr[period - 1] = _mean(tr[:period])
    for i in range(period, tr.size):
        atr[i] = ((atr[i - 1] * n_1) + tr[i]) / period
    return atr, atr[-1]


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _atr_update(
    atr_latest: f8,
    tr_current: f8,
    period: i4 = 14,
    n_1=13,
) -> f8:
    return ((atr_latest * n_1) + tr_current) / period


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _dm_smoothing_update(
    x: f8,
    s_p: f8,
    period: i4 = 14,
) -> f8:
    return s_p - (s_p / period) + x


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _dmi_update(
    dm: f8,
    tr: f8,
) -> f8:
    return (dm / tr) * 100


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _dx_update(
    pdmi: f8,
    ndmi: f8,
) -> f8:
    delta = pdmi - ndmi
    if delta == 0:
        return 0
    return (abs(pdmi - ndmi) / (pdmi + ndmi)) * 100


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
def _adx_update(
    dx: f8,
    adx_p: f8,
    adx_period: i4 = 14,
    n_1: id = 13,
) -> f8:
    return ((adx_p * n_1) + dx) / adx_period


@nb.njit(
    parallel=True,
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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

    # Reuse arrays to avoid multiple memory allocations
    tenkan_highs, tenkan_lows = _highs_lows(high, low, tenkan_period, to_range)
    kijun_highs, kijun_lows = _highs_lows(high, low, kijun_period, to_range)
    senkou_highs, senkou_lows = _highs_lows(high, low, senkou_period, to_range)

    # Reuse arrays for calculated outputs
    tenkan = (tenkan_highs + tenkan_lows) * 0.5
    kijun = (kijun_highs + kijun_lows) * 0.5
    senkou_b = (senkou_highs + senkou_lows) * 0.5

    # Initialize senkou_a for later use
    senkou_a = np.zeros(n, dtype=np.float64)

    # Compute senkou_a values from a_start
    for i in nb.prange(a_start, n):
        senkou_a[i] = (tenkan[i] + kijun[i]) * 0.5

    # Fill in missing values at the beginning with appropriate start values
    senkou_a[:a_start] = senkou_a[a_start]
    senkou_b[: senkou_period - 1] = senkou_b[senkou_period - 1]

    # Find the greatest period for later range slicing
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


@nb.njit(
    cache=NUMBA_DISK_CACHING,
    fastmath=True,
)
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
