# Candle Processing Architecture

## Overview

This document describes the SOLID-compliant architecture for processing market data ticks into OHLCV candles and computing indicators in parallel.

## Component Responsibilities (SRP)

| Component | Single Responsibility | Location |
|-----------|----------------------|----------|
| **Tick** | Represents a single price update | `src/ta/temporal.rs` |
| **Timeframe** | Defines candle duration (M1, M5, H1, etc.) | `src/ta/temporal.rs` |
| **CandleBuilder** | Aggregates ticks → OHLCV candles | `src/ta/temporal.rs` |
| **RollingWindow** | Stores candles with fixed capacity | `src/ta/math/rolling.rs` |
| **SharedWindow** | Thread-safe wrapper (`Arc<RwLock<RollingWindow>>`) | `src/ta/math/rolling.rs` |
| **Indicator** | Computes values from `&[Ohlcv]` slices | `src/ta/mod.rs` |

## Data Flow

```
                                    ┌─────────────────────────────────────┐
                                    │         Parallel Indicators         │
                                    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
                                    │  │SMA-7│ │SMA14│ │ EMA │ │ RSI │   │
                                    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
                                    │     │      │      │      │        │
                                    │     └──────┴──────┴──────┘        │
                                    │              │                     │
                                    │        par_iter()                  │
                                    └──────────────┬─────────────────────┘
                                                   │
                                            snapshot()
                                                   │
┌──────────┐      ┌───────────────┐      ┌────────┴────────┐
│  Ticks   │─────▶│ CandleBuilder │─────▶│  SharedWindow   │
│ (Market) │      │  (Timeframe)  │      │ Arc<RwLock<RW>> │
└──────────┘      └───────────────┘      └─────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  current()  │ ◀── Incomplete candle for real-time display
                  └─────────────┘
```

## Component Details

### Tick

Smallest unit of market data before aggregation.

```rust
pub struct Tick {
    pub timestamp: i64,  // Unix seconds
    pub price: f64,
    pub volume: f64,
}

// Create a tick
let tick = Tick::new(1703260800, 42150.50, 1.5);
```

### Timeframe

Defines candle duration. Minimum is 1 minute (no sub-minute).

```rust
pub enum Timeframe {
    M1, M5, M15, M30,           // Minutes
    H1, H2, H4, H8, H12,        // Hours
    D1, D3, W1, MN1,            // Days/Weeks/Months
}

let tf = Timeframe::M5;
tf.as_seconds()        // 300
tf.period_start(325)   // 300 (period boundary)
tf.is_new_period(299, 300)  // true
```

### CandleBuilder

Aggregates ticks into OHLCV candles based on timeframe boundaries.

```rust
let mut builder = CandleBuilder::new(Timeframe::M1);

// Process ticks - returns Some(candle) when period boundary crossed
for tick in ticks {
    if let Some(completed_candle) = builder.push(&tick) {
        window.push(completed_candle);
    }
}

// Access incomplete candle for real-time display
if let Some(current) = builder.current() {
    println!("Live: {}", current.close.0);
}

// Force-complete at market close
if let Some(last) = builder.flush() {
    window.push(last);
}
```

### SharedWindow

Thread-safe candle storage for parallel indicator access.

```rust
use std::sync::{Arc, RwLock};
use rolling_ta::ta::math::RollingWindow;

// Type alias (defined in rolling.rs)
pub type SharedWindow = Arc<RwLock<RollingWindow>>;

// Create shared window
let window: SharedWindow = Arc::new(RwLock::new(RollingWindow::new(200)));

// Push candles (write lock)
window.write().unwrap().push(candle);

// Take snapshot for indicators (read lock)
let snapshot: Vec<Ohlcv> = window.read().unwrap().snapshot();
```

### Indicator Trait

All indicators implement this trait for both batch and streaming modes.

```rust
pub trait Indicator: Send + Sync {
    type Output: Clone + Send;
    type Config: Clone + Default;

    // Batch: process all historical data
    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self>;

    // Streaming: process snapshot, return latest value
    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output>;

    fn latest(&self) -> Option<Self::Output>;
    fn history(&self) -> &[f64];
    fn reset(&mut self);
    fn warmup_period(&self) -> usize;
}
```

## Parallel Processing with Rayon

```rust
use rayon::prelude::*;

// Take snapshot once
let snapshot = Arc::new(window.read().unwrap().snapshot());

// Process multiple indicators in parallel
let configs = vec![("SMA-7", 7), ("SMA-14", 14), ("SMA-21", 21)];

let results: Vec<_> = configs
    .par_iter()
    .map(|(name, period)| {
        let mut sma = SMA::new(SMAConfig::new(*period));
        sma.calc(&snapshot).unwrap();
        (*name, sma.history().to_vec())
    })
    .collect();
```

## Multi-Timeframe Aggregation

Aggregate 1m candles to higher timeframes:

```rust
let mut builder_1m = CandleBuilder::new(Timeframe::M1);
let mut builder_5m = CandleBuilder::new(Timeframe::M5);
let mut builder_15m = CandleBuilder::new(Timeframe::M15);

for tick in ticks {
    // 1m aggregation
    if let Some(candle_1m) = builder_1m.push(&tick) {
        window_1m.push(candle_1m);

        // Use 1m close as input for higher timeframes
        let higher_tf_tick = Tick::new(
            candle_1m.timestamp.0,
            candle_1m.close.0,
            candle_1m.volume.0,
        );

        if let Some(candle_5m) = builder_5m.push(&higher_tf_tick) {
            window_5m.push(candle_5m);
        }
        if let Some(candle_15m) = builder_15m.push(&higher_tf_tick) {
            window_15m.push(candle_15m);
        }
    }
}
```

## Real-Time Streaming Pattern

```rust
// Shared state
let window: SharedWindow = Arc::new(RwLock::new(RollingWindow::new(200)));
let mut builder = CandleBuilder::new(Timeframe::M1);
let mut indicators = vec![
    SMA::new(SMAConfig::new(14)),
    SMA::new(SMAConfig::new(21)),
];

// On each tick from market feed
fn on_tick(tick: Tick, builder: &mut CandleBuilder, window: &SharedWindow, indicators: &mut [SMA]) {
    if let Some(completed) = builder.push(&tick) {
        // Push to shared window
        window.write().unwrap().push(completed);

        // Update all indicators with new snapshot
        let snapshot = window.read().unwrap().snapshot();
        for indicator in indicators.iter_mut() {
            if let Some(value) = indicator.next(&snapshot) {
                println!("New value: {}", value);
            }
        }
    }
}
```

## Indicator Migration Checklist

When migrating an indicator to the new architecture:

### 1. Remove Internal RollingWindow

```rust
// BEFORE (old)
pub struct EMA {
    window: RollingWindow,  // ❌ Remove this
    history: Vec<f64>,
}

// AFTER (new)
pub struct EMA {
    history: Vec<f64>,      // ✅ Just history
    latest: Option<f64>,
}
```

### 2. Implement `calc(&[Ohlcv])`

Batch calculation from slice:

```rust
fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
    self.reset();
    self.history.clear();

    let period = self.config.period;

    // Warmup: push NaN
    for _ in 0..period.saturating_sub(1) {
        self.history.push(f64::NAN);
    }

    // Compute values
    for i in (period - 1)..data.len() {
        let value = self.compute(&data[..=i]);
        self.history.push(value);
    }

    self.latest = self.history.last().copied();
    Ok(self)
}
```

### 3. Implement `next(&[Ohlcv])`

Streaming calculation from snapshot:

```rust
fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
    let period = self.config.period;

    // Need at least `period` candles
    if candles.len() < period {
        return None;
    }

    // Compute from last `period` candles
    let value = self.compute_from_slice(candles, period);

    self.latest = Some(value);
    self.history.push(value);

    Some(value)
}
```

### 4. Ensure Send + Sync

Indicators must be thread-safe for parallel processing:

```rust
// Derive or implement
#[derive(Debug, Clone)]
pub struct EMA {
    // All fields must be Send + Sync
    config: EMAConfig,
    state: IndicatorState,
    history: Vec<f64>,
    latest: Option<f64>,
}
// Vec<f64>, Option<f64>, etc. are automatically Send + Sync
```

### 5. Remove `update()` Method

The old `update(&Ohlcv)` method is deprecated. Use `next(&[Ohlcv])` instead:

```rust
// OLD (deprecated)
fn update(&mut self, candle: &Ohlcv) -> Option<f64>;

// NEW
fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output>;
```

### 6. Add Integration Test

Add test in `tests/rust/<indicator>.rs`:

```rust
#[test]
fn ema_batch_vs_reference() {
    let cols = read_xlsx_by_position("resources/data/btc-ema.xlsx", &[1, 2]);
    let candles = build_candles_from_closes(&cols[0]);
    let expected = &cols[1];

    let mut ema = EMA::new(EMAConfig::new(14));
    ema.calc(&candles).unwrap();

    compare_values("EMA", ema.history(), expected, EPSILON);
}

#[test]
fn ema_streaming_matches_batch() {
    // Same as sma_streaming_next_vs_batch pattern
}
```

## File Reference

| File | Purpose |
|------|---------|
| `src/ta/temporal.rs` | Tick, Timeframe, CandleBuilder |
| `src/ta/math/rolling.rs` | RollingWindow, SharedWindow |
| `src/ta/mod.rs` | Indicator trait |
| `src/ta/trend/sma.rs` | Reference implementation |
| `tests/rust/candles.rs` | Integration tests |
| `tests/rust/common.rs` | Test utilities |

## Key Invariants

1. **CandleBuilder is stateless for storage** - Only holds current incomplete candle
2. **Indicators compute from slices** - No internal window ownership
3. **SharedWindow is the single source of truth** - All indicators read from same snapshot
4. **`next()` only pushes to history when returning Some** - No NaN during warmup
5. **`calc()` includes NaN for warmup period** - History length matches input length