# Remove OhlcvSeries - Unify on &[Ohlcv]

## Motivation

`OhlcvSeries` (struct-of-arrays) exists only for `calc()` and tests. Runtime uses `&[Ohlcv]` (array-of-structs) from RollingWindow. Maintaining two data formats adds complexity without clear benefit.

**Principle:** Tests should mirror runtime. One data format everywhere.

## Current State

```
Testing:  XLSX → OhlcvSeries → calc()
Runtime:  DB/WS → &[Ohlcv] → next()
```

## Target State

```
Testing:  XLSX → Vec<Ohlcv> → calc() & next()
Runtime:  DB/WS → &[Ohlcv] → calc() & next()
```

## Files Affected

### Source Files (18)

| File | Change Required |
|------|-----------------|
| `src/ta/types.rs` | Remove `OhlcvSeries` struct |
| `src/ta/mod.rs` | Change `calc(&OhlcvSeries)` → `calc(&[Ohlcv])` |
| `src/ta/trend/sma.rs` | Update calc() signature and impl |
| `src/ta/trend/ema.rs` | Update calc() signature and impl |
| `src/ta/trend/wma.rs` | Update calc() signature and impl |
| `src/ta/trend/hma.rs` | Update calc() signature and impl |
| `src/ta/trend/macd.rs` | Update calc() signature and impl |
| `src/ta/trend/lr.rs` | Update calc() signature and impl |
| `src/ta/trend/adx.rs` | Update calc() signature and impl |
| `src/ta/trend/dmi.rs` | Update calc() signature and impl |
| `src/ta/momentum/rsi.rs` | Update calc() signature and impl |
| `src/ta/momentum/roc.rs` | Update calc() signature and impl |
| `src/ta/volatility/atr.rs` | Update calc() signature and impl |
| `src/ta/volatility/bb.rs` | Update calc() signature and impl |
| `src/ta/volatility/tr.rs` | Update calc() signature and impl |
| `src/ta/volume/obv.rs` | Update calc() signature and impl |
| `src/ta/volume/vwap.rs` | Update calc() signature and impl |
| `src/lib.rs` | Remove OhlcvSeries re-export |

### Test Files (2)

| File | Change Required |
|------|-----------------|
| `tests/rust/common.rs` | Replace `build_ohlcv_*` → `load_ohlcv_*` returning `Vec<Ohlcv>` |
| `tests/REFACTOR_TRACKING.md` | Update documentation |

## Implementation Plan

### Phase 1: Add Ohlcv Helpers

**File: `src/ta/types.rs`**

Add helper methods to work with `&[Ohlcv]`:

```rust
impl Ohlcv {
    /// Extract closes from slice
    pub fn closes(candles: &[Ohlcv]) -> Vec<f64> {
        candles.iter().map(|c| c.close.0).collect()
    }

    /// Extract highs from slice
    pub fn highs(candles: &[Ohlcv]) -> Vec<f64> {
        candles.iter().map(|c| c.high.0).collect()
    }

    /// Extract lows from slice
    pub fn lows(candles: &[Ohlcv]) -> Vec<f64> {
        candles.iter().map(|c| c.low.0).collect()
    }

    /// Extract volumes from slice
    pub fn volumes(candles: &[Ohlcv]) -> Vec<f64> {
        candles.iter().map(|c| c.volume.0).collect()
    }
}
```

### Phase 2: Update Indicator Trait

**File: `src/ta/mod.rs`**

```rust
pub trait Indicator: Send + Sync {
    type Output: Clone + Send;
    type Config: Clone + Default;

    fn state(&self) -> IndicatorState;

    /// Batch calculation from candle slice.
    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self>;

    /// Streaming calculation from snapshot.
    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output>;

    fn latest(&self) -> Option<Self::Output>;
    fn reset(&mut self);
    fn warmup_period(&self) -> usize;
}
```

### Phase 3: Migrate Indicators (One at a Time)

**Order by complexity (simplest first):**

1. **SMA** - simplest, only uses closes
2. **EMA** - similar to SMA
3. **WMA** - similar to SMA
4. **ROC** - only uses closes
5. **RSI** - only uses closes
6. **OBV** - uses closes + volumes
7. **TR** - uses high, low, close
8. **ATR** - uses TR
9. **BB** - uses closes + SMA
10. **VWAP** - uses high, low, close, volume
11. **DMI** - uses high, low, close
12. **ADX** - uses DMI (composite)
13. **HMA** - uses WMA (composite)
14. **MACD** - uses EMA (composite)
15. **LR** - uses typical price

**Migration pattern per indicator:**

```rust
// Before
fn calc(&mut self, data: &OhlcvSeries) -> TAResult<&mut Self> {
    let closes = &data.closes;
    // ...
}

// After
fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
    let closes: Vec<f64> = data.iter().map(|c| c.close.0).collect();
    // ... or use Ohlcv::closes(data) helper
}
```

### Phase 4: Update Test Utilities

**File: `tests/rust/common.rs`**

Replace OhlcvSeries builders with Vec<Ohlcv> loaders:

```rust
// Before
pub fn build_ohlcv_from_closes(closes: &[f64]) -> OhlcvSeries

// After
pub fn build_candles_from_closes(closes: &[f64]) -> Vec<Ohlcv> {
    closes.iter().enumerate().map(|(i, &close)| {
        Ohlcv::new(i as i64, close, close, close, close, 0.0)
    }).collect()
}

pub fn build_candles_hlc(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<Ohlcv> {
    highs.iter().zip(lows).zip(closes).enumerate()
        .map(|(i, ((&h, &l), &c))| {
            Ohlcv::new(i as i64, c, h, l, c, 0.0)
        }).collect()
}

// ... similar for other builders
```

### Phase 5: Update Tests

**All test files in `tests/rust/*.rs`:**

```rust
// Before
let data = build_ohlcv_from_closes(closes);
sma.calc(&data).unwrap();

// After
let candles = build_candles_from_closes(closes);
sma.calc(&candles).unwrap();
```

### Phase 6: Remove OhlcvSeries

**File: `src/ta/types.rs`**

Delete:
- `OhlcvSeries` struct
- `OhlcvSeries::with_capacity()`
- `OhlcvSeries::from_closes()`
- `OhlcvSeries::from_tuples()`
- `OhlcvSeries::len()`
- `OhlcvSeries::is_empty()`
- `OhlcvSeries::push()`
- `OhlcvSeries::get()`

**File: `src/lib.rs`**

Remove `OhlcvSeries` from prelude/exports.

### Phase 7: Update shared_rolling_window.md

Update Data Format Rationale diagram to show unified format.

## Migration Checklist

### Phase 1: Helpers
- [ ] Add `Ohlcv::closes()` helper
- [ ] Add `Ohlcv::highs()` helper
- [ ] Add `Ohlcv::lows()` helper
- [ ] Add `Ohlcv::volumes()` helper

### Phase 2: Trait
- [ ] Update `Indicator::calc()` signature in `src/ta/mod.rs`

### Phase 3: Indicators
- [ ] SMA
- [ ] EMA
- [ ] WMA
- [ ] ROC
- [ ] RSI
- [ ] OBV
- [ ] TR
- [ ] ATR
- [ ] BB
- [ ] VWAP
- [ ] DMI
- [ ] ADX
- [ ] HMA
- [ ] MACD
- [ ] LR

### Phase 4: Test Utilities
- [ ] Replace `build_ohlcv_from_closes` → `build_candles_from_closes`
- [ ] Replace `build_ohlcv_hlc` → `build_candles_hlc`
- [ ] Replace `build_ohlcv_cv` → `build_candles_cv`
- [ ] Replace `build_ohlcv_thlcv` → `build_candles_thlcv`
- [ ] Update `tick_at()` if needed
- [ ] Update `slice_ohlcv()` → works with `&[Ohlcv]` already

### Phase 5: Tests
- [ ] Update all indicator tests to use new builders

### Phase 6: Cleanup
- [ ] Remove `OhlcvSeries` from `src/ta/types.rs`
- [ ] Remove from `src/lib.rs` exports
- [ ] Remove from prelude

### Phase 7: Documentation
- [ ] Update `shared_rolling_window.md`
- [ ] Update `REFACTOR_TRACKING.md`

## Verification

After each indicator migration:
```bash
cargo test
cargo clippy
```

After full migration:
```bash
cargo test --all
cargo clippy --all-targets
```

## Rollback Plan

If issues arise:
1. Each phase is independent
2. Git revert individual commits
3. OhlcvSeries can coexist during migration (deprecate first, remove later)