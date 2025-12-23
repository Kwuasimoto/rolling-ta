# rolling-ta - Technical Analysis Library

High-performance Rust library for computing technical indicators with streaming support.

## Core Concepts

### Data Types

All market data uses NewType wrappers for type safety:

```rust
use rolling_ta::prelude::*;

// Ohlcv is the primary candle type
let candle = Ohlcv::new(
    timestamp,  // i64 (Unix seconds)
    open,       // f64
    high,       // f64
    low,        // f64
    close,      // f64
    volume,     // f64
);

// Quick construction for testing
let candle = Ohlcv::from_close(100.0);

// Extract fields from slices
let closes: Vec<f64> = Ohlcv::closes(&candles);
let highs: Vec<f64> = Ohlcv::highs(&candles);
```

### Indicator Trait

All indicators implement `Indicator` for unified batch/streaming usage:

```rust
pub trait Indicator: Send + Sync {
    type Output: Clone + Send;
    type Config: Clone + Default;

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self>;  // Batch
    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output>;  // Streaming
    fn latest(&self) -> Option<Self::Output>;
    fn reset(&mut self);
    fn warmup_period(&self) -> usize;
}
```

Indicators with history storage implement `HistoricalIndicator`:

```rust
pub trait HistoricalIndicator: Indicator {
    fn history(&self) -> &[Self::Output];
    fn get(&self, index: isize) -> Option<Self::Output>;  // Supports negative indexing
    fn len(&self) -> usize;
}
```

## Usage Patterns

### Batch Mode (Historical Data)

Process all historical data at once:

```rust
use rolling_ta::trend::{SMA, SMAConfig};
use rolling_ta::prelude::*;

let candles: Vec<Ohlcv> = load_historical_data();

let mut sma = SMA::new(SMAConfig::new(14));
sma.calc(&candles)?;

// Access results
let all_values = sma.history();      // Full history with NaN warmup
let latest = sma.latest();           // Most recent value
let specific = sma.get(-1);          // Last value (negative indexing)
```

### Streaming Mode (Real-Time)

Process snapshots from a shared window:

```rust
use rolling_ta::ta::math::{SharedWindow, RollingWindow};
use std::sync::{Arc, RwLock};

// Create shared window
let window: SharedWindow = Arc::new(RwLock::new(RollingWindow::new(200)));

// Push completed candles
window.write().unwrap().push(candle);

// Take snapshot for indicators (parallel-safe)
let snapshot = window.read().unwrap().snapshot();

// Update indicator with snapshot
let mut sma = SMA::new(SMAConfig::new(14));
if let Some(value) = sma.next(&snapshot) {
    println!("SMA: {}", value);
}
```

### Tick-to-Candle Aggregation

Aggregate raw ticks into OHLCV candles:

```rust
use rolling_ta::ta::temporal::{CandleBuilder, Tick, Timeframe};

let mut builder = CandleBuilder::new(Timeframe::M1);

// Process ticks - returns Some(candle) when period boundary crossed
for tick in market_feed {
    if let Some(completed) = builder.push(&tick) {
        window.write().unwrap().push(completed);
    }
}

// Access incomplete candle for real-time display
if let Some(current) = builder.current() {
    println!("Live: {}", current.close.0);
}

// Force-complete at market close
if let Some(last) = builder.flush() {
    window.write().unwrap().push(last);
}
```

### Parallel Computation with Rayon

```rust
use rayon::prelude::*;

// Snapshot once, process in parallel
let snapshot = Arc::new(window.read().unwrap().snapshot());

let results: Vec<_> = [7, 14, 21]
    .par_iter()
    .map(|&period| {
        let mut sma = SMA::new(SMAConfig::new(period));
        sma.calc(&snapshot).unwrap();
        (period, sma.history().to_vec())
    })
    .collect();
```

## Available Indicators

### Trend (`rolling_ta::trend`)
| Indicator | Output | Config |
|-----------|--------|--------|
| `SMA` | `f64` | `SMAConfig::new(period)` |
| `EMA` | `f64` | `EMAConfig::new(period)` |
| `WMA` | `f64` | `WMAConfig::new(period)` |
| `HMA` | `f64` | `HMAConfig::new(period)` |
| `MACD` | `MACDOutput` | `MACDConfig::new(fast, slow, signal)` |
| `ADX` | `ADXOutput` | `ADXConfig::new(period)` |
| `DMI` | `DMIOutput` | `DMIConfig::new(period)` |
| `Ichimoku` | `IchimokuOutput` | `IchimokuConfig::new(...)` |
| `LR` | `LROutput` | `LRConfig::new(period)` |

### Momentum (`rolling_ta::momentum`)
| Indicator | Output | Config |
|-----------|--------|--------|
| `RSI` | `f64` | `RSIConfig::new(period)` |
| `ROC` | `f64` | `ROCConfig::new(period)` |
| `StochRSI` | `StochRSIOutput` | `StochRSIConfig::new(...)` |
| `BOP` | `f64` | `BOPConfig::default()` |

### Volatility (`rolling_ta::volatility`)
| Indicator | Output | Config |
|-----------|--------|--------|
| `ATR` | `f64` | `ATRConfig::new(period)` |
| `BB` | `BBOutput` | `BBConfig::new(period, std_dev)` |
| `Donchian` | `DonchianOutput` | `DonchianConfig::new(period)` |
| `TR` | `f64` | `TRConfig::default()` |

### Volume (`rolling_ta::volume`)
| Indicator | Output | Config |
|-----------|--------|--------|
| `OBV` | `f64` | `OBVConfig::default()` |
| `VWAP` | `f64` | `VWAPConfig::default()` |
| `MFI` | `f64` | `MFIConfig::new(period)` |
| `CMF` | `f64` | `CMFConfig::new(period)` |

## Architecture Reference

```
Tick → CandleBuilder → SharedWindow ← snapshot() ← Indicator
         ↓                   ↓
      current()         par_iter()
         ↓                   ↓
   (live display)    (parallel indicators)
```

Key invariants:
1. **Indicators compute from `&[Ohlcv]` slices** - no internal window ownership
2. **SharedWindow is single source of truth** - all indicators read from same snapshot
3. **`calc()` includes NaN for warmup** - history length matches input length
4. **`next()` only pushes when returning Some** - no NaN during streaming warmup
5. **All indicators are `Send + Sync`** - safe for parallel computation

## Example: Reference SMA Implementation

```rust
impl Indicator for SMA {
    type Output = f64;
    type Config = SMAConfig;

    fn calc(&mut self, data: &[Ohlcv]) -> TAResult<&mut Self> {
        // Fill NaN for warmup period
        for _ in 0..(period - 1) {
            self.history.push(f64::NAN);
        }
        // Compute values
        for i in (period - 1)..data.len() {
            let sum: f64 = data[i+1-period..=i].iter().map(|c| c.close.0).sum();
            self.history.push(sum / period as f64);
        }
        self.state = IndicatorState::Ready;
        Ok(self)
    }

    fn next(&mut self, candles: &[Ohlcv]) -> Option<Self::Output> {
        if candles.len() < self.config.period {
            return None;  // Warmup
        }
        let sma = Self::compute_from_slice(candles, self.config.period);
        self.history.push(sma);
        Some(sma)
    }
}
```

## Error Handling

```rust
use rolling_ta::ta::error::{TAError, TAResult};

let result: TAResult<_> = sma.calc(&candles);
match result {
    Ok(indicator) => { /* use indicator */ }
    Err(TAError::InsufficientData { required, actual }) => {
        eprintln!("Need {} candles, got {}", required, actual);
    }
    Err(TAError::InvalidPeriod(p)) => {
        eprintln!("Invalid period: {}", p);
    }
    _ => {}
}
```

## Timeframes

```rust
use rolling_ta::ta::temporal::Timeframe;

Timeframe::M1   // 1 minute (60s)
Timeframe::M5   // 5 minutes (300s)
Timeframe::M15  // 15 minutes (900s)
Timeframe::H1   // 1 hour (3600s)
Timeframe::D1   // 1 day (86400s)

// Convert from seconds
let tf = Timeframe::from_seconds(300);  // Some(Timeframe::M5)

// Period boundaries
tf.period_start(325);       // 300
tf.is_new_period(299, 300); // true
```