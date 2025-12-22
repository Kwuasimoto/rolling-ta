# Indicator Migration Plan

## Overview

Migrate all indicators from old architecture (OhlcvSeries, update(), internal Temporal) to new architecture (&[Ohlcv] slices, next(), CandleBuilder for temporal).

Reference architecture: `.claude/arch/candle_processing.md`

## Migration Status

### Trend Indicators

| Indicator | Status | Notes |
|-----------|--------|-------|
| SMA | Done | Reference implementation |
| EMA | Done | Catch-up logic for late joining snapshots |
| WMA | Done | Uses last N candles (no catch-up needed) |
| HMA | Done | Computes WMA internally from slice |
| MACD | Done | Composite of EMA, signal line warmup |
| DMI | Done | Wilder smoothing for +DM/-DM/TR, committed state |
| ADX | Done | Composite of DMI, Wilder smoothing for ADX |
| LR | Done | LinearRegression, LinearRegressionR2, LinearRegressionForecast |

### Momentum Indicators

| Indicator | Status | Notes |
|-----------|--------|-------|
| RSI | Done | Uses Wilder smoothing, streaming warmup tracking |
| ROC | Done | Simple lookback, computes directly from slice |
| StochRSI | Done | Composite of RSI + Stochastic, outputs %K and %D |
| BOP | Done | (Close-Open)/(High-Low) with SMA smoothing |

### Volatility Indicators

| Indicator | Status | Notes |
|-----------|--------|-------|
| TR | Done | True Range, foundation for ATR |
| ATR | Done | Uses Wilder smoothing |
| BB | Done | Bollinger Bands, computes mean+stddev directly |
| Donchian | Done | High/Low channels, computes directly from slice |

### Volume Indicators

| Indicator | Status | Notes |
|-----------|--------|-------|
| OBV | Done | Cumulative volume, committed state for same-candle updates |
| VWAP | Done | Cumulative with time-based reset, dual accumulators |
| MFI | Done | Uses rolling window of money flows, committed state |
| CMF | Done | Uses MFM × Volume, rolling windows for sums |

### Other

| Indicator | Status | Notes |
|-----------|--------|-------|
| Ichimoku | Done | Tenkan/Kijun/Senkou A/B, window-based (no committed state) |

## Migration Checklist (per indicator)

- [ ] Remove internal RollingWindow/Temporal
- [ ] Remove timeframe from Config (if present)
- [ ] Change `calc(&OhlcvSeries)` → `calc(&[Ohlcv])`
- [ ] Remove `update(&Ohlcv)`
- [ ] Add `next(&[Ohlcv]) -> Option<Output>`
- [ ] Ensure `Send + Sync` (for Rayon)
- [ ] Update tests to use `&[Ohlcv]`
- [ ] Add integration test in `tests/rust/<indicator>.rs`
- [ ] Enable in `src/ta/trend/mod.rs` (or appropriate module)

## Config Changes Required

### EMAConfig
```rust
// BEFORE
pub struct EMAConfig {
    pub period: usize,
    pub timeframe: i64,  // REMOVE
}

// AFTER
pub struct EMAConfig {
    pub period: usize,
}
```

### WMAConfig
```rust
// BEFORE
pub struct WMAConfig {
    pub period: usize,
    pub timeframe: i64,  // REMOVE
}

// AFTER
pub struct WMAConfig {
    pub period: usize,
}
```

## EMA-Specific Notes

EMA requires special handling for streaming:
- Keep `prev_ema` state for incremental calculation
- First EMA value is seeded with SMA
- Multiplier = 2 / (period + 1)
- `next()` needs to track whether we're continuing or restarting

### EMA next() Strategy

For `next(&[Ohlcv])`:
1. If no prev_ema: compute initial SMA from last `period` candles
2. If has prev_ema: apply EMA step to latest close
3. Track `last_len` to detect if snapshot grew (new candle) vs same snapshot

## Testing Strategy

1. **Batch vs Reference**: Compare `calc()` output to Python reference (xlsx)
2. **Batch vs Streaming**: Compare `calc()` output to sequential `next()` calls
3. **Parallel Safety**: Verify indicators work with `Arc<Vec<Ohlcv>>` and Rayon

## Priority Order

1. **EMA** - Foundation for MACD
2. **WMA** - Foundation for HMA
3. **RSI** - Popular momentum indicator
4. **ATR** - Uses Wilder (similar to EMA)
5. **BB** - Uses SMA
6. **MACD** - Uses EMA
7. **HMA** - Uses WMA
8. Remaining indicators

## Files to Update

For each indicator:
- `src/ta/<category>/<indicator>.rs` - Implementation
- `src/ta/<category>/mod.rs` - Enable exports
- `src/ta/config.rs` - Remove timeframe if present
- `tests/rust/<indicator>.rs` - Integration test
- `tests/rust.rs` - Enable test module