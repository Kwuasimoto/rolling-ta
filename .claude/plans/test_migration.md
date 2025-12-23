# Test Migration Plan

## Overview

Migrate all tests in `tests/rust/` from old architecture (`update(&Tick)`) to new architecture (`calc(&[Ohlcv])`, `next(&[Ohlcv])`).

Reference implementations: `sma.rs`, `ema.rs`, `wma.rs`, `hma.rs` (already migrated)

---

## Current State

### Enabled Tests (Already Migrated)

| Test | Status | Pattern |
|------|--------|---------|
| sma.rs | Done | batch vs reference, streaming vs batch, fixed window |
| ema.rs | Done | batch vs reference, streaming vs batch, fixed window |
| wma.rs | Done | batch vs reference, streaming vs batch, fixed window |
| hma.rs | Done | batch vs reference, streaming vs batch, fixed window |
| candles.rs | Done | CandleBuilder, SharedWindow tests |

### Disabled Tests (Need Migration)

| Test | XLSX Data | Old Pattern | Notes |
|------|-----------|-------------|-------|
| rsi.rs | btc-rsi.xlsx | `update(&tick_at())` | Single output |
| bb.rs | btc-bb.xlsx | `update(&tick_at())` | Compound: upper, middle, lower |
| atr.rs | btc-atr.xlsx | `update(&tick_at())` | Also tests TR |
| obv.rs | btc-obv.xlsx | `update(&tick_at())` | Cumulative indicator |
| adx.rs | btc-adx.xlsx | `update(&tick_at())` | Compound: +DI, -DI, DX, ADX |
| vwap.rs | btc-vwap.xlsx | `update(&tick_at())` | Cumulative with reset |
| roc.rs | btc-roc.xlsx | `update(&tick_at())` | Simple lookback |
| lr.rs | btc-linear_regression.xlsx | `update(&tick_at())` | LR, LR2, LRF variants |

### Missing Tests (Need Creation)

| Indicator | XLSX Data | Notes |
|-----------|-----------|-------|
| StochRSI | btc-stoch_rsi.xlsx | Compound: %K, %D |
| BOP | btc-bop.xlsx | Single output |
| CMF | btc-cmf.xlsx | Single output |
| MFI | btc-mfi.xlsx | Single output |
| Donchian | btc-donchian.xlsx | Compound: upper, middle, lower |
| Ichimoku | btc-ichimoku_cloud.xlsx | Compound: tenkan, kijun, senkou_a, senkou_b |
| DMI | (no xlsx) | Compound: +DI, -DI; batch vs streaming only | *NOTE: USE ADX DATA,

---

## Migration Checklist (Per Test)

### Pattern Change

**Old (remove):**
```rust
// Streaming via update()
for i in 0..data.len() {
    indicator.update(&tick_at(&data, i)).unwrap();
}

// Hybrid mode
indicator.calc(&slice_ohlcv(&data, 0, half)).unwrap();
for i in half..data.len() {
    indicator.update(&tick_at(&data, i)).unwrap();
}
```

**New (use):**
```rust
// Streaming via next() with growing snapshot
for i in 1..=candles.len() {
    let snapshot = &candles[..i];
    indicator.next(snapshot);
}

// Or fixed window pattern
for i in period..candles.len() {
    let snapshot = &candles[i - period + 1..=i];
    indicator.next(snapshot);
}
```

### Standard Test Structure

Each migrated test should have:

1. **`*_batch_vs_reference`** - Compare `calc()` output to xlsx data
2. **`*_streaming_next_vs_batch`** - Compare `next()` with growing snapshots to `calc()`
3. **`*_next_with_fixed_window`** - Compare fixed-window `next()` to xlsx data

---

## common.rs Updates

### Remove (Old Architecture)

```rust
// DELETE these functions:
fn tick_at(data: &[Ohlcv], i: usize) -> Tick { ... }
fn slice_ohlcv(data: &[Ohlcv], start: usize, end: usize) -> Vec<Ohlcv> { ... }
fn build_ohlcv_from_closes(closes: &[f64]) -> Vec<Ohlcv> { ... }  // rename
fn build_ohlcv_hlc(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<Ohlcv> { ... }  // rename
```

### Keep/Update

```rust
// KEEP these functions:
fn read_xlsx_by_position(...) -> Vec<Vec<f64>>
fn build_candles_from_closes(closes: &[f64]) -> Vec<Ohlcv>  // existing
fn build_candles_ohlcv(...) -> Vec<Ohlcv>  // existing
fn compare_values(...) -> usize
fn assert_histories_equal(...)
```

### Add (Compound Output Helpers)

```rust
fn assert_bb_histories_equal(context: &str, a: &[BBOutput], b: &[BBOutput], epsilon: f64)
fn assert_adx_histories_equal(context: &str, a: &[ADXOutput], b: &[ADXOutput], epsilon: f64)
fn assert_ichimoku_histories_equal(context: &str, a: &[IchimokuOutput], b: &[IchimokuOutput], epsilon: f64)
fn assert_stochrsi_histories_equal(context: &str, a: &[StochRSIOutput], b: &[StochRSIOutput], epsilon: f64)
fn assert_donchian_histories_equal(context: &str, a: &[DonchianOutput], b: &[DonchianOutput], epsilon: f64)
fn assert_dmi_histories_equal(context: &str, a: &[DMIOutput], b: &[DMIOutput], epsilon: f64)
```

---

## XLSX Column Reference

From xlsx files (NO headers, 0-indexed):

| File | Columns |
|------|---------|
| btc-sma.xlsx | ts(0), close(1), sma(2) |
| btc-ema.xlsx | ts(0), close(1), ema(2) |
| btc-wma.xlsx | ts(0), close(1), wma(2) |
| btc-hma.xlsx | ts(0), close(1), hma(2) |
| btc-rsi.xlsx | ts(0), close(1), ..., rsi(6) |
| btc-roc.xlsx | ts(0), close(1), roc(2) |
| btc-bb.xlsx | ts(0), close(1), sma(2), upper(3), lower(4) |
| btc-atr.xlsx | ts(0), high(1), low(2), close(3), tr(4), atr(5) |
| btc-adx.xlsx | ts(0), high(1), low(2), close(3), ..., +dmi(12), -dmi(13), dx(14), adx(15) |
| btc-obv.xlsx | ts(0), close(1), volume(2), obv(3) |
| btc-vwap.xlsx | ts(0), high(1), low(2), close(3), volume(4), vwap(5) |
| btc-linear_regression.xlsx | ts(0), close(1), lr(2), lr_r2(3), lr_forecast(4) |
| btc-stoch_rsi.xlsx | (check columns) |
| btc-bop.xlsx | (check columns) |
| btc-cmf.xlsx | (check columns) |
| btc-mfi.xlsx | (check columns) |
| btc-donchian.xlsx | (check columns) |
| btc-ichimoku_cloud.xlsx | (check columns) |

---

## Migration Order

### Phase 1: Update common.rs
- [ ] Remove old helper functions (tick_at, slice_ohlcv)
- [ ] Rename build_ohlcv_* to build_candles_* for consistency
- [ ] Add compound output comparison helpers

### Phase 2: Simple Output Indicators
- [ ] rsi.rs - Wilder smoothing, single output
- [ ] roc.rs - Simple lookback, single output
- [ ] obv.rs - Cumulative, single output

### Phase 3: Compound Output Indicators
- [ ] bb.rs - upper/middle/lower
- [ ] atr.rs - TR + ATR
- [ ] adx.rs - +DI/-DI/DX/ADX
- [ ] vwap.rs - single output but cumulative
- [ ] lr.rs - LR/LR2/LRF variants

### Phase 4: New Tests
- [ ] stoch_rsi.rs - %K/%D compound output
- [ ] bop.rs - single output
- [ ] cmf.rs - single output
- [ ] mfi.rs - single output
- [ ] donchian.rs - upper/middle/lower
- [ ] ichimoku.rs - tenkan/kijun/senkou_a/senkou_b
- [ ] dmi.rs - +DI/-DI (batch vs streaming only, no xlsx)

### Phase 5: Enable in rust.rs
- [ ] Uncomment migrated test modules
- [ ] Add new test modules
- [ ] Verify all tests pass

---

## Success Criteria

1. All tests use `calc(&[Ohlcv])` and `next(&[Ohlcv])` pattern
2. No references to `update()` or `Tick` in test code
3. All tests compare against xlsx reference data (where available)
4. Batch vs streaming equivalence verified for all indicators
5. `cargo test` passes with all modules enabled