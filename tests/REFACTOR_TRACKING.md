# Reference Tests Refactoring Tracker

Two-phase refactoring:
1. Consolidate 3 tests per indicator into 1 `*_modes_equivalent` test
2. Split monolithic `reference_tests.rs` into individual files under `tests/rust/`

## New Structure

```
tests/
  rust.rs                    # Crate root, declares modules with #[path] attrs
  rust/
    common.rs                # Shared test utilities
    sma.rs                   # SMA tests (done)
    ema.rs                   # EMA tests (pending)
    ...
  reference_tests.rs         # OLD - to be deleted when migration complete
```

## Test Pattern

Replace:
- `*_matches_reference` (batch vs Python)
- `*_streaming_matches_batch` (stream vs batch)
- `*_hybrid_mode` (50% batch + 50% stream vs batch)

With single:
- `*_modes_equivalent` (all three modes in one test)

## Migration Status

| Indicator | Consolidated | Migrated | Temporalized | Notes |
|-----------|--------------|----------|--------------|-------|
| SMA | Yes | Yes | Yes | `rust/sma.rs` |
| EMA | Yes | Yes | Yes | `rust/ema.rs` |
| WMA | Yes | Yes | Yes | `rust/wma.rs` |
| HMA | Yes | Yes | No | `rust/hma.rs` |
| RSI | Yes | Yes | No | `rust/rsi.rs` |
| BB | Yes | Yes | No | `rust/bb.rs` (compound: upper/middle/lower) |
| ATR | Yes | Yes | No | `rust/atr.rs` (includes TR validation) |
| OBV | Yes | Yes | No | `rust/obv.rs` (uses relative error for large values) |
| ADX | Yes | Yes | No | `rust/adx.rs` (compound: plus_di/minus_di/dx/adx) |
| LinearRegression | Yes | Yes | No | `rust/lr.rs` (LR, LR2, LRF + optimized_path) |
| VWAP | Yes | Yes | No | `rust/vwap.rs` (uses timestamps for reset interval) |
| ROC | Yes | Yes | No | `rust/roc.rs` |

### Temporalization

Temporalized indicators support real-time tick aggregation via `Config::with_timeframe(period, timeframe)`.
When enabled, multiple ticks within the same candle period update the latest value in place
instead of pushing new history entries.

## Shared Helpers (tests/rust/common.rs)

```rust
// Data loading
fn read_xlsx_by_position(path: &str, col_indices: &[usize]) -> Vec<Vec<f64>>

// OHLCV builders
fn build_ohlcv_from_closes(closes: &[f64]) -> OhlcvSeries
fn build_ohlcv_hlc(highs: &[f64], lows: &[f64], closes: &[f64]) -> OhlcvSeries
fn build_ohlcv_cv(closes: &[f64], volumes: &[f64]) -> OhlcvSeries
fn build_ohlcv_thlcv(...) -> OhlcvSeries

// Tick/slice helpers
fn tick_at(data: &OhlcvSeries, i: usize) -> Ohlcv
fn slice_ohlcv(data: &OhlcvSeries, start: usize, end: usize) -> OhlcvSeries

// Comparison helpers
fn compare_values(name: &str, rust: &[f64], expected: &[f64], epsilon: f64) -> usize
fn assert_histories_equal(context: &str, a: &[f64], b: &[f64], epsilon: f64)
```

## TODO: Compound Output Helpers

For indicators with multi-field outputs (BB, ADX), add to common.rs:

```rust
fn assert_bb_histories_equal(context: &str, a: &[BBOutput], b: &[BBOutput], epsilon: f64)
fn assert_adx_histories_equal(context: &str, a: &[ADXOutput], b: &[ADXOutput], epsilon: f64)
```

## Cleanup

Migration complete:
1. ~~Delete `tests/reference_tests.rs`~~ DONE
2. ~~Remove duplicate SMA tests from old file~~ DONE

All 12 indicators consolidated and migrated to `tests/rust/` structure.
