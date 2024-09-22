import numba as nb
import numpy as np

from numba.types import f8, f4, i8, i4

from rolling_ta.env import (
    NUMBA_DISK_CACHING,
    NUMBA_PARALLEL,
    NUMBA_NOGIL,
    NUMBA_FASTMATH,
)
from rolling_ta.logging import logger

logger.debug(
    f"Numba environment: [Caching={NUMBA_DISK_CACHING}, Parallel={NUMBA_PARALLEL}, Nogil={NUMBA_NOGIL}, Numba={NUMBA_FASTMATH}]"
)

## // HELPER FUNCTIONS \\


@nb.njit(
    parallel=NUMBA_PARALLEL,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _highs_lows(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    highs_container: np.ndarray[f8],
    lows_container: np.ndarray[f8],
    period: i4,
    to_range: i4,
):
    for i in nb.prange(period, to_range):
        max_high = np.max(high[i - period : i])  # Use vectorized np.max
        min_low = np.min(low[i - period : i])  # Use vectorized np.min
        highs_container[i - 1] = max_high
        lows_container[i - 1] = min_low


@nb.njit(
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _prefix_sum(arr: np.ndarray[f8]) -> np.ndarray[f8]:
    n = arr.size
    prefix_sum = arr.copy()
    prefix_sum[0] = arr[0]
    for i in range(1, n):
        prefix_sum[i] = arr[i] + prefix_sum[i - 1]
    return prefix_sum


@nb.njit(
    parallel=NUMBA_PARALLEL,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _mean(arr: np.ndarray[f8]) -> f8:
    n = arr.size
    sum: f8 = 0.0
    for i in nb.prange(n):
        sum += arr[i]
    return sum / n


@nb.njit(
    parallel=NUMBA_PARALLEL,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _sliding_midpoint(
    high: np.ndarray[f8], low: np.ndarray[f8], x_container: np.ndarray[f8], period: i4
):
    n: i8 = high.size

    for i in nb.prange(period - 1, n):
        max_val: f8 = -np.inf
        min_val: f8 = np.inf

        for j in range(i - period + 1, i + 1):
            if high[j] > max_val:
                max_val = high[j]
            if low[j] < min_val:
                min_val = low[j]

        x_container[i] = (max_val + min_val) * 0.5


@nb.njit(
    parallel=NUMBA_PARALLEL,
    inline="always",
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
)
def _typical_price(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    close: np.ndarray[f8],
    typical_price_container: np.ndarray[f8],
):
    for i in nb.prange(typical_price_container.size):
        typical_price_container[i] = _typical_price_single(high[i], low[i], close[i])


@nb.njit(inline="always", cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH)
def _typical_price_single(high: f8, low: f8, close: f8) -> f8:
    return (high + low + close) / 3


## // INDICATOR FUNCTIONS \\


@nb.njit(parallel=NUMBA_PARALLEL, cache=NUMBA_DISK_CACHING, nogil=NUMBA_NOGIL)
def _sma(
    data: np.ndarray[f8],
    sma_container: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], np.ndarray[f8], f8, f8]:
    current_sum: f8 = 0.0
    for i in nb.prange(period):
        current_sum += data[i]
    sma_container[period - 1] = current_sum / period
    for i in range(period, data.size):
        current_sum += data[i] - data[i - period]
        sma_container[i] = current_sum / period
    return sma_container, data[-period:], current_sum, sma_container[-1]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _sma_update(
    close: f8, window_sum: f8, window: np.ndarray[f8], period: i4 = 14
) -> tuple[np.ndarray[f8], f8, f8]:
    first = window[0]
    window[:-1] = window[1:]
    window[-1] = close
    window_sum = (window_sum - first) + close
    return window_sum / period, window, window_sum


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _ema(
    data: np.ndarray[f8], ema_container: np.ndarray[f8], weight: f8, period: i4 = 14
) -> tuple[np.ndarray[f8], f8]:
    current_sum = 0.0
    for i in nb.prange(period):
        current_sum += data[i]
    ema_prev = current_sum / period
    ema_container[period - 1] = ema_prev
    for i in range(period, data.shape[0]):
        ema_prev = ((data[i] - ema_prev) * weight) + ema_prev
        ema_container[i] = ema_prev
    return ema_container, ema_container[-1]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _ema_update(close: f8, weight: f8, ema_latest: f8) -> np.ndarray[f8]:
    return ((close - ema_latest) * weight) + ema_latest


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
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


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _rsi_update(
    close: f8, prev_close: f8, avg_gain: f8, avg_loss: f8, p_1: f8 = 13
) -> tuple[f8, f8, f8]:
    delta = close - prev_close

    gain = max(delta, 0)
    loss = -min(delta, 0)

    avg_gain = p_1 * (gain - avg_gain) + avg_gain
    avg_loss = p_1 * (loss - avg_loss) + avg_loss

    rsi = (100 * avg_gain) / (avg_gain + avg_loss)

    return rsi, avg_gain, avg_loss


@nb.njit(parallel=NUMBA_PARALLEL, cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH)
def _stoch_rsi(
    rsi: np.ndarray[f8],
    window: np.ndarray[f8],
    stoch_rsi_container: np.ndarray[f8],
    rsi_period: i4 = 14,
    stoch_period: i4 = 10,
) -> None:
    n = rsi.size

    for i in nb.prange(rsi_period, n - stoch_period):
        curr_rsi = rsi[i]
        y = i + stoch_period

        max_rsi = max(rsi[i:y])
        min_rsi = min(rsi[i:y])
        stoch_rsi_container[i] = (curr_rsi - min_rsi) / (max_rsi - min_rsi)


@nb.njit(parallel=NUMBA_PARALLEL, cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH)
def _stoch_k(stoch_rsi: np.ndarray[f8], k_container: np.ndarray[f8], k_period: i4):
    for i in nb.prange():
        pass


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _obv(
    close: np.ndarray[f8], volume: np.ndarray[f8], obv_container: np.ndarray[f8]
) -> tuple[np.ndarray[f8], f8, f8]:
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


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _obv_update(close: f8, volume: f8, close_p: f8, obv_latest: f8) -> f8:
    if close > close_p:
        obv_latest += volume
    elif close < close_p:
        obv_latest -= volume
    return obv_latest


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _rmf(price: np.ndarray[f8], volume: np.ndarray[f8], rmf_container: np.ndarray[f8]):
    for i in nb.prange(price.size):
        rmf_container[i] = price[i] * volume[i]


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _mf_pos_neg(
    price: np.ndarray[f8],
    rmf: np.ndarray[f8],
    pmf_container: np.ndarray[f8],
    nmf_container: np.ndarray[f8],
):
    for i in nb.prange(1, price.size):
        curr_typical = price[i]
        prev_typical = price[i - 1]

        if curr_typical > prev_typical:
            pmf_container[i] = rmf[i]
        elif curr_typical < prev_typical:
            nmf_container[i] = rmf[i]


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _mf_pos_neg_sum(
    pmf: np.ndarray[f8],
    nmf: np.ndarray[f8],
    pmf_sum_container: np.ndarray[f8],
    nmf_sum_container: np.ndarray[f8],
    period: i8 = 14,
):
    for i in nb.prange(period, pmf.size):
        pmf_sum: f8 = 0.0
        nmf_sum: f8 = 0.0

        for j in nb.prange((i - period) + 1, i + 1):
            pmf_sum += pmf[j]
            nmf_sum += nmf[j]

        pmf_sum_container[i] = pmf_sum
        nmf_sum_container[i] = nmf_sum


# njit caching is breaking this single function? Runs once but then dies trying
# to load from cache
@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _mfi(
    pmf_sums: np.ndarray[f8],
    nmf_sums: np.ndarray[f8],
    mfi_container: np.ndarray[f8],
    mfi_period: i8 = 14,
) -> tuple[np.ndarray[f8], np.ndarray[f8]]:

    for i in nb.prange(mfi_period, mfi_container.size):
        pmf_sum = pmf_sums[i]
        nmf_sum = nmf_sums[i]
        mfi_container[i] = (100 * pmf_sum) / (pmf_sum + nmf_sum)


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _mf_update(
    volume: f8,
    price_curr: f8,
    price_prev: f8,
    pmf_window: np.ndarray[f8],
    nmf_window: np.ndarray[f8],
    pmf_sum: f8,
    nmf_sum: f8,
) -> tuple[f8, f8]:
    rmf = volume * price_curr

    pmf_popped: f8 = pmf_window[0]
    nmf_popped: f8 = nmf_window[0]
    pmf_window[:-1] = pmf_window[1:]
    nmf_window[:-1] = nmf_window[1:]

    if price_curr > price_prev:
        pmf_window[-1] = rmf
        nmf_window[-1] = 0
        pmf_sum += rmf
    elif price_curr < price_prev:
        pmf_window[-1] = 0
        nmf_window[-1] = rmf
        nmf_sum += rmf
    else:
        pmf_window[-1] = 0
        nmf_window[-1] = 0

    pmf_sum -= pmf_popped
    nmf_sum -= nmf_popped

    return pmf_sum, nmf_sum


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _mfi_update(pmf_sum: f8, nmf_sum: f8) -> tuple[f8, f8, f8]:
    return (100 * pmf_sum) / (pmf_sum + nmf_sum)


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _tr(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    close: np.ndarray[f8],
    close_p_container: np.ndarray[f8],
    tr_container: np.ndarray[f8],
) -> tuple[np.ndarray[f8], f8, f8]:
    n = close.size

    close_p_container[1:] = close[:-1]
    tr_container[0] = high[0] - low[0]

    for i in nb.prange(1, n):
        tr_container[i] = max(
            high[i] - low[i],
            abs(high[i] - close_p_container[i]),
            close_p_container[i] - low[i],
        )

    return tr_container, tr_container[-1], close[-1]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _tr_update(high: f8, low: f8, close_p: f8) -> f8:
    return max(high - low, abs(high - close_p), close_p - low)


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _atr(
    tr: np.ndarray[f8], atr_container: np.ndarray[f8], period: i4 = 14, n_1: i4 = 13
) -> tuple[np.ndarray[f8], f8]:
    atr_container[period - 1] = _mean(tr[:period])
    for i in range(period, tr.size):
        atr_container[i] = ((atr_container[i - 1] * n_1) + tr[i]) / period
    return atr_container, atr_container[-1]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _atr_update(atr_latest: f8, tr_current: f8, period: i4 = 14, n_1=13) -> f8:
    return ((atr_latest * n_1) + tr_current) / period


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _dm(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    high_p_container: np.ndarray[f8],
    low_p_container: np.ndarray[f8],
) -> tuple[np.ndarray[f8], np.ndarray[f8], f8, f8]:

    high_p_container[1:] = high[:-1]
    low_p_container[1:] = low[:-1]

    high[0] = 0.0
    low[0] = 0.0
    high_p_container[0] = 0.0
    low_p_container[0] = 0.0

    move_up = high - high_p_container
    move_down = low_p_container - low

    move_up_mask = (move_up > 0) & (move_up > move_down)
    move_down_mask = (move_down > 0) & (move_down > move_up)

    pdm = np.zeros(move_up_mask.size, dtype=np.float64)
    ndm = np.zeros(move_down_mask.size, dtype=np.float64)

    pdm[move_up_mask] = move_up[move_up_mask]
    ndm[move_down_mask] = move_down[move_down_mask]

    return pdm, ndm, high[-1], low[-1]


# OK
@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _dm_update(high: f8, low: f8, high_p: f8, low_p: f8) -> tuple[f8, f8]:
    move_up = high - high_p
    move_down = low_p - low

    move_up_mask = (move_up > 0) & (move_up > move_down)
    move_down_mask = (move_down > 0) & (move_down > move_up)

    pdm = move_up if move_up_mask else 0.0
    ndm = move_down if move_down_mask else 0.0

    return pdm, ndm


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _dm_smoothing(
    x: np.ndarray[f8], s_x_container: np.ndarray[f8], period: i4 = 14
) -> tuple[np.ndarray[f8], f8]:
    # According to: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx
    # The initial TrueRange value (ex: high - low) is not a valid True Range, so we start at period + 1
    p_1 = period + 1
    x_sum: i8 = 0

    for i in nb.prange(1, p_1):
        x_sum += x[i]
    s_x_container[period] = x_sum

    for i in range(p_1, x.size):
        s_x_p = s_x_container[i - 1]
        s_x_container[i] = s_x_p - (s_x_p / period) + x[i]

    return s_x_container, s_x_container[-1]


# OK
@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _dm_smoothing_update(x: f8, s_x_p: f8, period: i4 = 14) -> f8:
    return s_x_p - (s_x_p / period) + x


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _dmi(
    dm: np.ndarray[f8],
    tr: np.ndarray[f8],
    dmi_container: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    n = dm.size

    for i in nb.prange(period, n):
        dmi_container[i] = (dm[i] / tr[i]) * 100

    return dmi_container, dmi_container[-1]


# OK
@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _dmi_update(s_dm: f8, s_tr: f8) -> f8:
    return (s_dm / s_tr) * 100


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _dx(
    pdmi: np.ndarray[f8],
    ndmi: np.ndarray[f8],
    dx_container: np.ndarray[f8],
    period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    n = pdmi.size

    for i in nb.prange(period, n):
        if pdmi[i] + ndmi[i] != 0:
            dx_container[i] = (abs(pdmi[i] - ndmi[i]) / (pdmi[i] + ndmi[i])) * 100
        else:
            dx_container[i] = 0

    return dx_container, dx_container[-1]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _dx_update(pdmi: f8, ndmi: f8) -> f8:
    delta = pdmi - ndmi
    if delta == 0:
        return 0
    return (abs(pdmi - ndmi) / (pdmi + ndmi)) * 100


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _adx(
    dx: np.ndarray[f8],
    adx_container: np.ndarray[f8],
    adx_period: i4 = 14,
    dmi_period: i4 = 14,
) -> tuple[np.ndarray[f8], f8]:
    pp: i4 = adx_period + dmi_period
    weight: i4 = adx_period - 1
    adx_0: f8 = 0.0

    for i in nb.prange(adx_period, pp):
        adx_0 += dx[i]

    adx_container[pp - 1] = adx_0 / adx_period

    for i in range(pp, adx_container.size):
        adx_container[i] = ((adx_container[i - 1] * weight) + dx[i]) / adx_period

    return adx_container, adx_container[-1]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _adx_update(
    dx: f8,
    adx_p: f8,
    adx_period: i4 = 14,
    n_1: id = 13,
) -> f8:
    return ((adx_p * n_1) + dx) / adx_period


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _tenkan(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    tenkan_container: np.ndarray[f8],
    tenkan_period: f8,
):
    _sliding_midpoint(high, low, tenkan_container, tenkan_period)


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _tenkan_update(
    high_container: np.ndarray[f8],
    low_container: np.ndarray[f8],
    tenkan_period: i8,
) -> f8:
    return (
        max(high_container[-tenkan_period:]) + min(low_container[-tenkan_period:])
    ) * 0.5


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _kijun(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    kijun_container: np.ndarray[f8],
    kijun_period: f8,
):
    _sliding_midpoint(high, low, kijun_container, kijun_period)


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _kijun_update(
    high_container: np.ndarray[f8],
    low_container: np.ndarray[f8],
    kijun_period: i8,
) -> f8:
    return (
        max(high_container[-kijun_period:]) + min(low_container[-kijun_period:])
    ) * 0.5


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _senkou_b(
    high: np.ndarray[f8],
    low: np.ndarray[f8],
    senkou_b_container: np.ndarray[f8],
    senkou_period: f8,
):
    _sliding_midpoint(high, low, senkou_b_container, senkou_period)
    senkou_b_container[: senkou_period - 1] = senkou_b_container[senkou_period - 1]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _senkou_b_update(
    high_container: np.ndarray[f8],
    low_container: np.ndarray[f8],
    senkou_period: i8,
) -> f8:
    return (
        max(high_container[-senkou_period:]) + min(low_container[-senkou_period:])
    ) * 0.5


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _senkou_a(
    tenkan: np.ndarray[f8],
    kijun: np.ndarray[f8],
    senkou_a_container: np.ndarray[f8],
    tenkan_period: f8,
    kijun_period: f8,
):
    a_start = max(tenkan_period, kijun_period) - 1
    for i in nb.prange(a_start, tenkan.size):
        senkou_a_container[i] = (tenkan[i] + kijun[i]) * 0.5
    senkou_a_container[:a_start] = senkou_a_container[a_start]


@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _senkou_a_update(tenkan: f8, kijun: f8) -> f8:
    return (tenkan + kijun) * 0.5


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _linear_regression(
    ys: np.ndarray[f8],
    slope_container: np.ndarray[f8],
    intercept_container: np.ndarray[f8],
    period: i4 = 14,
):
    n: i8 = ys.size
    p_1: i8 = period - 1

    x: i4 = 0
    xx: i8 = 0.0

    for i in nb.prange(period):
        x += i
        xx += i * i

    for i in nb.prange(p_1, n):
        y: f8 = 0.0
        xy: f8 = 0.0

        for j in nb.prange(i - p_1, i + 1):
            y += ys[j]
            xy += (j - (i - p_1)) * ys[j]

        slope_container[i] = ((period * xy) - (x * y)) / ((period * xx) - (x * x))
        intercept_container[i] = (y - (slope_container[i] * x)) / period


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _linear_regression_forecast(
    slopes: np.ndarray[f8],
    intercepts: np.ndarray[f8],
    forecast_container: np.ndarray[f8],
    forecast: i4 = 14,
):
    n = slopes.size
    assert forecast_container.size >= n + forecast

    # Calculate the initial forecast, using constant 1
    for i in nb.prange(slopes.size):
        forecast_container[i] = (slopes[i] * 1) + intercepts[i]

    # We only use the last slop and intercept to calculate the forecast?
    slope = slopes[-1]
    intercept = intercepts[-1]

    for i in nb.prange(1, forecast + 1):
        j = i + n - 1
        # We add +1 to i because if we don't, we get the value forecast_container[-1].
        forecast_container[j] = (slope * (i + 1)) + intercept


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_DISK_CACHING,
    fastmath=NUMBA_FASTMATH,
    nogil=NUMBA_NOGIL,
)
def _linear_regression_r2(
    ys: np.ndarray[f8],
    slopes: np.ndarray[f8],
    intercepts: np.ndarray[f8],
    r2_container: np.ndarray[f8],
    period: i4 = 14,
):
    n: i4 = ys.size
    p_1: i4 = period - 1

    # Start from the period-1, loop to size of ys.
    for i in nb.prange(p_1, n):

        # Calculate y_mean, start from 0, to 13 (14 iterations)
        y_mean: f8 = 0.0
        for j in range(i - p_1, i + 1):
            y_mean += ys[j]
        y_mean /= period

        ss_r: f8 = 0.0
        ss_t: f8 = 0.0

        for k in range(period):
            y = ys[k + (i - p_1)]
            p = slopes[i] * k + intercepts[i]

            p_delta = y - p
            m_delta = y - y_mean

            ss_r += p_delta * p_delta
            ss_t += m_delta * m_delta

        if ss_t > 0.0:
            r2_container[i] = 1 - (ss_r / ss_t)
        else:
            r2_container[i] = 1.0
