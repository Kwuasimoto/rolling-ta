# Rust Translation Plan: rolling-ta → Asterion TA Module

> **Status**: Phase 3 Complete (7/7 simple indicators) — Reference tests verified ✅
> **Created**: 2025-12-10
> **Pattern**: Strategy + Builder + State Machine

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture Analysis](#architecture-analysis)
- [Recommended Simplifications](#recommended-simplifications)
- [Module Structure](#module-structure)
- [Core Trait Design](#core-trait-design)
- [Implementation Phases](#implementation-phases)
- [Numba → Rust Translation Guide](#numba--rust-translation-guide)
- [Integration with Asterion Pipeline](#integration-with-asterion-pipeline)
- [Type System Design](#type-system-design)
- [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Current Python Architecture (rolling-ta)

| Aspect | Python Implementation | Verdict |
|--------|----------------------|---------|
| Base class inheritance | `Indicator` ABC with `calc()` + `update()` | **Keep** — Clean pattern |
| Numba JIT functions | Stateless math, state passed as params | **Keep** — Already Rust-friendly |
| Composition pattern | Indicators nest other indicators | **Simplify** — Use owned fields |
| String-keyed outputs | `_keys = ["sma"]`, `to_numpy(get="sma")` | **Replace** — Use enums/structs |
| Dict period config | `_period_config = {"sma": 14}` | **Replace** — Use struct fields |
| Optional memory storage | `_memory: bool`, `array.array` | **Simplify** — Always store or use ring buffer |
| Initialization state | `_initialized: bool` + propagation | **Replace** — State enum |

### Recommendation: Partial Simplification

The Python architecture is **well-designed** but uses dynamic typing idioms. Rust benefits from:

1. **Static types** over string keys
2. **Owned state** over shared mutable references
3. **Result types** over silent failures
4. **Builder pattern** for configuration (replaces dict config)

---

## Architecture Analysis

### What Works Well (Keep)

#### 1. Two-Phase Calculation Pattern

```python
# Python: calc() for batch, update() for streaming
indicator.calc()           # Process historical data
indicator.update(new_row)  # Process single tick
```

**Rust equivalent:**
```rust
impl Indicator for SMA {
    fn calc(&mut self, data: &OhlcvSeries) -> Result<(), TAError>;
    fn update(&mut self, tick: &Ohlcv) -> Result<f64, TAError>;
}
```

#### 2. Stateless Math Functions

The Numba functions are **pure** — they take state as input and return new state:

```python
# Python: No side effects, explicit state
def _sma_update(close: f8, window_sum: f8, window: np.ndarray, period: i4):
    # Returns: (new_value, updated_window, updated_sum)
```

**Rust equivalent:**
```rust
#[inline]
fn sma_step(close: f64, window: &mut VecDeque<f64>, sum: &mut f64) -> f64 {
    let old = window.pop_front().unwrap_or(0.0);
    window.push_back(close);
    *sum = *sum - old + close;
    *sum / window.len() as f64
}
```

#### 3. Composition Over Inheritance

Complex indicators (BB, ADX, HMA) compose simpler ones. This maps directly to Rust:

```rust
pub struct BollingerBands {
    sma: SMA,  // Owned dependency
    config: BBConfig,
    state: BBState,
}
```

### What Needs Simplification

#### 1. String-Keyed Outputs → Structs

**Python (dynamic):**
```python
_keys = ["upper", "lower", "ma"]
def to_numpy(self, get: str = "ma"):
    match get:
        case "upper": return self._upper
        case "lower": return self._lower
        case "ma": return self._ma.to_numpy()
```

**Rust (static):**
```rust
pub struct BBOutput {
    pub upper: f64,
    pub lower: f64,
    pub ma: f64,
}

impl BollingerBands {
    pub fn latest(&self) -> BBOutput { ... }
    pub fn history(&self) -> &[BBOutput] { ... }
}
```

#### 2. Dict Period Config → Builder Pattern

**Python:**
```python
_period_config = {"bb": 20, "ma": 20, "weight": 2.0}
bb = BollingerBands(period_config={"bb": 14, "weight": 1.5})
```

**Rust:**
```rust
pub struct BBConfig {
    pub period: usize,
    pub ma_period: usize,
    pub std_dev_weight: f64,
}

impl Default for BBConfig {
    fn default() -> Self {
        Self { period: 20, ma_period: 20, std_dev_weight: 2.0 }
    }
}

// Usage:
let bb = BollingerBands::new(BBConfig { period: 14, ..Default::default() });
```

#### 3. Initialization State → State Enum

**Python:**
```python
_initialized: bool = False
_init: bool = False  # confusing naming

def calc(self):
    if self._initialized and not force:
        return
```

**Rust:**
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndicatorState {
    Uninitialized,
    Warming(usize),  // ticks processed < period
    Ready,
}

impl SMA {
    pub fn state(&self) -> IndicatorState { self.state }

    pub fn update(&mut self, tick: &Ohlcv) -> Result<Option<f64>, TAError> {
        match self.state {
            IndicatorState::Uninitialized => Err(TAError::NotInitialized),
            IndicatorState::Warming(n) if n < self.config.period => {
                // Return None while warming
                Ok(None)
            }
            IndicatorState::Ready => {
                Ok(Some(self.compute_step(tick.close)))
            }
        }
    }
}
```

#### 4. Optional Memory → Ring Buffer

**Python:**
```python
_memory: bool = True
_retention: Optional[int] = None

if self._memory:
    self._sma = array('f', sma)
```

**Rust:**
```rust
use circular_buffer::CircularBuffer;

pub struct SMA {
    config: SMAConfig,
    state: SMAState,
    // Always store, but bounded
    history: CircularBuffer<1024, f64>,  // Fixed capacity
}

// Or use VecDeque with max_capacity
```

---

## Module Structure

```
src-tauri/src/
├── ta/
│   ├── mod.rs              # Re-exports + Indicator trait
│   ├── error.rs            # TAError enum
│   ├── types.rs            # Ohlcv, OhlcvSeries, common types
│   ├── config.rs           # Config structs for all indicators
│   ├── state.rs            # IndicatorState enum + state structs
│   ├── math/
│   │   ├── mod.rs
│   │   ├── rolling.rs      # Rolling window helpers
│   │   ├── ema.rs          # EMA math (no state)
│   │   └── stats.rs        # Mean, variance, etc.
│   ├── trend/
│   │   ├── mod.rs
│   │   ├── sma.rs
│   │   ├── ema.rs
│   │   ├── wma.rs
│   │   ├── hma.rs
│   │   ├── macd.rs
│   │   └── adx.rs          # Composes DMI
│   ├── momentum/
│   │   ├── mod.rs
│   │   ├── rsi.rs
│   │   └── stoch_rsi.rs    # Composes RSI
│   ├── volatility/
│   │   ├── mod.rs
│   │   ├── tr.rs
│   │   ├── atr.rs          # Composes TR
│   │   └── bb.rs           # Composes SMA
│   └── volume/
│       ├── mod.rs
│       ├── obv.rs
│       ├── mfi.rs
│       └── vwap.rs
└── pipeline/
    └── stages/
        └── technical.rs    # Uses ta:: module
```

---

## Core Trait Design

### Primary Trait: `Indicator`

```rust
// src-tauri/src/ta/mod.rs

use crate::ta::error::TAError;
use crate::ta::types::{Ohlcv, OhlcvSeries};
use crate::ta::state::IndicatorState;

/// Core indicator trait.
///
/// # Contract
/// - `calc()` processes historical batch data, sets state to Ready
/// - `update()` processes single ticks, requires Ready state
/// - `reset()` returns to Uninitialized state
pub trait Indicator: Send + Sync {
    /// Output type (single value or struct of values)
    type Output: Clone + Send;

    /// Configuration type
    type Config: Clone + Default;

    /// Current state (Uninitialized, Warming, Ready)
    fn state(&self) -> IndicatorState;

    /// Batch calculation over historical data
    fn calc(&mut self, data: &OhlcvSeries) -> Result<(), TAError>;

    /// Single-tick update (streaming)
    /// Returns None if still warming up, Some(output) if ready
    fn update(&mut self, tick: &Ohlcv) -> Result<Option<Self::Output>, TAError>;

    /// Get latest calculated value (None if not ready)
    fn latest(&self) -> Option<Self::Output>;

    /// Reset to uninitialized state
    fn reset(&mut self);

    /// Minimum data points required before output
    fn warmup_period(&self) -> usize;
}
```

### Secondary Trait: `HistoricalIndicator`

```rust
/// Extension trait for indicators that store history
pub trait HistoricalIndicator: Indicator {
    /// Get all calculated values
    fn history(&self) -> &[Self::Output];

    /// Get value at index (negative for from end)
    fn get(&self, index: isize) -> Option<Self::Output>;

    /// Number of stored values
    fn len(&self) -> usize;
}
```

### Why This Design?

| Python Feature | Rust Equivalent | Rationale |
|----------------|-----------------|-----------|
| `calc()` + `update()` | Same methods | Proven pattern, matches use case |
| `_initialized` bool | `IndicatorState` enum | Type-safe, explicit warmup |
| `to_numpy(get=...)` | Associated `Output` type | Static typing, no string keys |
| `_period_config` dict | Associated `Config` type | Compile-time validation |
| Optional history | `HistoricalIndicator` trait | Separation of concerns |

---

## Implementation Phases

### Phase 1: Foundation (Core Infrastructure) ✅ COMPLETE

- [x] `ta/error.rs` — TAError enum
- [x] `ta/types.rs` — Ohlcv, OhlcvSeries, Price, Volume newtypes
- [x] `ta/state.rs` — IndicatorState enum
- [x] `ta/config.rs` — Config structs for all indicators
- [x] `ta/mod.rs` — Indicator trait + re-exports

### Phase 2: Math Layer (Pure Functions) ✅ COMPLETE

Translate numba.py functions to Rust. These are stateless helpers.

- [x] `ta/math/rolling.rs` — Rolling window utilities
- [x] `ta/math/ema.rs` — EMA multiplier calculation
- [x] `ta/math/stats.rs` — Mean, variance, std dev

### Phase 3: Simple Indicators ✅ COMPLETE

Indicators with single output, no dependencies.

| Priority | Indicator | Complexity | Status |
|----------|-----------|------------|--------|
| 1 | SMA | Low | ✅ Complete |
| 2 | EMA | Low | ✅ Complete |
| 3 | RSI | Medium | ✅ Complete |
| 4 | TR | Low | ✅ Complete |
| 5 | ATR | Medium | ✅ Complete |
| 6 | WMA | Medium | ✅ Complete |
| 7 | OBV | Low | ✅ Complete |

### Phase 4: Composite Indicators

Indicators that compose Phase 3 indicators.

| Priority | Indicator | Dependencies | Notes |
|----------|-----------|--------------|-------|
| 1 | BollingerBands | SMA | Std dev calculation |
| 2 | MACD | EMA x2 | Three outputs |
| 3 | HMA | WMA x2 | Complex composition |
| 4 | DMI | TR | Multiple outputs |
| 5 | ADX | DMI | Deep composition |
| 6 | StochRSI | RSI | Stochastic of RSI |

### Phase 5: Multi-Output Indicators

| Indicator | Outputs | Notes |
|-----------|---------|-------|
| Donchian | high, low, center | |
| Ichimoku | 5 lines | Most complex |
| VWAP | vwap | Session-based reset |
| MFI | mfi | Uses volume |

### Phase 6: Pipeline Integration

- [ ] Modify `TACalculator` in Asterion to use `ta::` module
- [ ] Wire indicator configs through Tauri state
- [ ] Add Tauri commands for indicator configuration

---

## Numba → Rust Translation Guide

### Pattern: Batch Function

**Python (numba.py):**
```python
@nb.njit(parallel=NUMBA_PARALLEL, cache=NUMBA_DISK_CACHING, ...)
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
```

**Rust:**
```rust
/// Calculate SMA over entire dataset
/// Returns: (output_vec, window_state, sum_state, latest_value)
pub fn sma_batch(
    data: &[f64],
    period: usize,
) -> (Vec<f64>, VecDeque<f64>, f64, f64) {
    let mut output = vec![f64::NAN; data.len()];
    let mut sum = 0.0;

    // Initial sum
    for i in 0..period {
        sum += data[i];
    }
    output[period - 1] = sum / period as f64;

    // Rolling calculation
    for i in period..data.len() {
        sum += data[i] - data[i - period];
        output[i] = sum / period as f64;
    }

    // Build window state for updates
    let window: VecDeque<f64> = data[data.len() - period..].iter().copied().collect();
    let latest = *output.last().unwrap_or(&f64::NAN);

    (output, window, sum, latest)
}
```

### Pattern: Update Function

**Python:**
```python
@nb.njit(cache=NUMBA_DISK_CACHING, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def _sma_update(
    close: f8, window_sum: f8, window: np.ndarray[f8], period: i4 = 14
) -> tuple[np.ndarray[f8], f8, f8]:
    first = window[0]
    window[:-1] = window[1:]
    window[-1] = close
    window_sum = (window_sum - first) + close
    return window_sum / period, window, window_sum
```

**Rust:**
```rust
/// Update SMA with single new value
/// Modifies window and sum in place, returns new SMA value
#[inline]
pub fn sma_update(
    close: f64,
    window: &mut VecDeque<f64>,
    sum: &mut f64,
) -> f64 {
    let old = window.pop_front().unwrap_or(0.0);
    window.push_back(close);
    *sum = *sum - old + close;
    *sum / window.len() as f64
}
```

### Translation Table: Numba → Rust

| Numba | Rust | Notes |
|-------|------|-------|
| `np.ndarray[f8]` | `&[f64]` or `Vec<f64>` | Use slice for input, Vec for output |
| `nb.prange(n)` | `(0..n)` | Rust ranges are lazy iterators |
| `parallel=True` | `rayon::par_iter()` | Optional, benchmark first |
| `cache=True` | N/A | Rust compiles AOT |
| `fastmath=True` | `#[target_feature(enable = "fma")]` | Optional, arch-specific |
| `f8`, `f4` | `f64`, `f32` | |
| `i8`, `i4` | `i64`, `i32` | Or `usize` for indices |
| `np.zeros(n)` | `vec![0.0; n]` | |
| `np.max(arr)` | `arr.iter().fold(f64::NEG_INFINITY, \|a, &b\| a.max(b))` | Or use `itertools::max` |
| `arr[:-1] = arr[1:]` | `window.pop_front(); window.push_back(v)` | VecDeque for sliding window |

### Complex Example: RSI

**Python:**
```python
@nb.njit(parallel=NUMBA_PARALLEL, ...)
def _rsi(close, rsi_container, gains_container, losses_container, period=14, p_1=13):
    n = close.size

    # Phase 1 (SMA)
    for i in nb.prange(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains_container[i] = delta
        elif delta < 0:
            losses_container[i] = -delta

    avg_gain = _mean(gains_container[1:period + 1])
    avg_loss = _mean(losses_container[1:period + 1])
    rsi_container[period] = (100 * avg_gain) / (avg_gain + avg_loss)

    # Phase 2 (EMA)
    for i in range(period + 1, n):
        avg_gain = ((avg_gain * p_1) + gains_container[i]) / period
        avg_loss = ((avg_loss * p_1) + losses_container[i]) / period
        rsi_container[i] = (100 * avg_gain) / (avg_gain + avg_loss)

    return rsi_container, avg_gain, avg_loss, close[-1]
```

**Rust:**
```rust
pub struct RSIState {
    pub avg_gain: f64,
    pub avg_loss: f64,
    pub prev_close: f64,
}

pub fn rsi_batch(close: &[f64], period: usize) -> (Vec<f64>, RSIState) {
    let n = close.len();
    let mut output = vec![f64::NAN; n];
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];
    let p_1 = (period - 1) as f64;

    // Calculate gains/losses
    for i in 1..n {
        let delta = close[i] - close[i - 1];
        if delta > 0.0 {
            gains[i] = delta;
        } else if delta < 0.0 {
            losses[i] = -delta;
        }
    }

    // Initial averages (SMA)
    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    output[period] = 100.0 * avg_gain / (avg_gain + avg_loss);

    // Smoothed averages (Wilder's smoothing)
    for i in (period + 1)..n {
        avg_gain = (avg_gain * p_1 + gains[i]) / period as f64;
        avg_loss = (avg_loss * p_1 + losses[i]) / period as f64;
        output[i] = 100.0 * avg_gain / (avg_gain + avg_loss);
    }

    let state = RSIState {
        avg_gain,
        avg_loss,
        prev_close: *close.last().unwrap(),
    };

    (output, state)
}

#[inline]
pub fn rsi_update(close: f64, state: &mut RSIState, period: usize) -> f64 {
    let delta = close - state.prev_close;
    let (gain, loss) = if delta > 0.0 {
        (delta, 0.0)
    } else {
        (0.0, -delta)
    };

    let p_1 = (period - 1) as f64;
    state.avg_gain = (state.avg_gain * p_1 + gain) / period as f64;
    state.avg_loss = (state.avg_loss * p_1 + loss) / period as f64;
    state.prev_close = close;

    100.0 * state.avg_gain / (state.avg_gain + state.avg_loss)
}
```

---

## Integration with Asterion Pipeline

### Current Pipeline (from asterion.md)

```
WebSocket → [OHLCV] → [TACalculator] → [Normalizer] → [ML] → [Action]
                           ↓
                         tap.send()
```

### Modified TACalculator

```rust
// src-tauri/src/pipeline/stages/technical.rs

use crate::ta::{
    Indicator,
    trend::{EMA, SMA},
    momentum::RSI,
    volatility::ATR,
};

pub struct TACalculator {
    tap_sender: broadcast::Sender<TABundle>,

    // Indicators as owned fields (composition)
    rsi: RSI,
    ema_fast: EMA,
    ema_slow: EMA,
    atr: ATR,

    tick_count: usize,
}

impl TACalculator {
    pub fn new(
        tap_sender: broadcast::Sender<TABundle>,
        config: TAConfig,
    ) -> Self {
        Self {
            tap_sender,
            rsi: RSI::new(config.rsi),
            ema_fast: EMA::new(config.ema_fast),
            ema_slow: EMA::new(config.ema_slow),
            atr: ATR::new(config.atr),
            tick_count: 0,
        }
    }
}

impl PipelineStage<Ohlcv, TABundle> for TACalculator {
    fn process(&mut self, input: Ohlcv) -> PipelineResult<TABundle> {
        self.tick_count += 1;

        // Update all indicators
        let rsi = self.rsi.update(&input)?;
        let ema_fast = self.ema_fast.update(&input)?;
        let ema_slow = self.ema_slow.update(&input)?;
        let atr = self.atr.update(&input)?;

        // Build MACD from EMAs (if both ready)
        let macd = match (ema_fast, ema_slow) {
            (Some(fast), Some(slow)) => Some(MacdValues {
                macd_line: fast - slow,
                signal_line: 0.0,  // TODO: signal EMA
                histogram: 0.0,
            }),
            _ => None,
        };

        let bundle = TABundle {
            timestamp: input.timestamp,
            rsi,
            macd,
            ema_fast: ema_fast.map(Price),
            ema_slow: ema_slow.map(Price),
            atr,
            source_ohlcv: input,
        };

        let _ = self.tap_sender.send(bundle.clone());
        Ok(bundle)
    }

    fn name(&self) -> &'static str { "TACalculator" }
    fn tap(&self) -> &broadcast::Sender<TABundle> { &self.tap_sender }
}
```

### Configuration via Tauri Commands

```rust
// src-tauri/src/commands/ta.rs

use crate::ta::config::*;

#[tauri::command]
pub async fn configure_indicators(
    state: State<'_, PipelineState>,
    rsi_period: Option<usize>,
    ema_fast_period: Option<usize>,
    ema_slow_period: Option<usize>,
) -> Result<(), String> {
    let mut pipeline = state.pipeline.lock().await;

    if let Some(period) = rsi_period {
        pipeline.ta_calculator.rsi = RSI::new(RSIConfig { period });
    }
    // ... etc

    Ok(())
}
```

---

## Type System Design

### NewTypes for Domain Safety

```rust
// src-tauri/src/ta/types.rs

/// Price value (prevents confusion with volume/count)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Price(pub f64);

/// Volume value
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Volume(pub f64);

/// Unix timestamp in seconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub i64);

/// OHLCV candle
#[derive(Debug, Clone)]
pub struct Ohlcv {
    pub timestamp: Timestamp,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Volume,
}

/// Historical OHLCV data
pub struct OhlcvSeries {
    pub timestamps: Vec<Timestamp>,
    pub opens: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub closes: Vec<f64>,
    pub volumes: Vec<f64>,
}
```

### Config Structs (Replace Dict)

```rust
// src-tauri/src/ta/config.rs

#[derive(Debug, Clone)]
pub struct SMAConfig {
    pub period: usize,
}

impl Default for SMAConfig {
    fn default() -> Self { Self { period: 14 } }
}

#[derive(Debug, Clone)]
pub struct RSIConfig {
    pub period: usize,
}

impl Default for RSIConfig {
    fn default() -> Self { Self { period: 14 } }
}

#[derive(Debug, Clone)]
pub struct BBConfig {
    pub period: usize,
    pub ma_period: usize,
    pub std_dev_weight: f64,
}

impl Default for BBConfig {
    fn default() -> Self {
        Self {
            period: 20,
            ma_period: 20,
            std_dev_weight: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MACDConfig {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
}

impl Default for MACDConfig {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}
```

### Output Structs (Replace String Keys)

```rust
// Each indicator defines its output type

/// Single value output (SMA, EMA, RSI, ATR)
pub type SingleOutput = f64;

/// Bollinger Bands output
#[derive(Debug, Clone, Copy)]
pub struct BBOutput {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

/// MACD output
#[derive(Debug, Clone, Copy)]
pub struct MACDOutput {
    pub macd: f64,
    pub signal: f64,
    pub histogram: f64,
}

/// DMI output
#[derive(Debug, Clone, Copy)]
pub struct DMIOutput {
    pub plus_di: f64,
    pub minus_di: f64,
    pub adx: f64,
}
```

---

## Testing Strategy

### Unit Tests: Math Functions

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sma_batch() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (output, _, _, latest) = sma_batch(&data, 3);

        assert!(output[0].is_nan());
        assert!(output[1].is_nan());
        assert_relative_eq!(output[2], 2.0, epsilon = 1e-10);  // (1+2+3)/3
        assert_relative_eq!(output[3], 3.0, epsilon = 1e-10);  // (2+3+4)/3
        assert_relative_eq!(latest, 9.0, epsilon = 1e-10);     // (8+9+10)/3
    }

    #[test]
    fn test_sma_update() {
        let mut window = VecDeque::from([8.0, 9.0, 10.0]);
        let mut sum = 27.0;

        let result = sma_update(11.0, &mut window, &mut sum);

        assert_relative_eq!(result, 10.0, epsilon = 1e-10);  // (9+10+11)/3
        assert_eq!(window.len(), 3);
        assert_eq!(*window.back().unwrap(), 11.0);
    }
}
```

### Integration Tests: Against Python Reference ✅ IMPLEMENTED

Reference tests read xlsx fixtures generated by Python rolling-ta and verify Rust outputs match.

**Test Results (all passing):**

| Indicator | Values Verified |
|-----------|-----------------|
| SMA       | 187             |
| EMA       | 187             |
| WMA       | 187             |
| RSI       | 186             |
| TR        | 200             |
| ATR       | 187             |
| OBV       | 200             |

```rust
// tests/reference_tests.rs (actual implementation)
use calamine::{open_workbook, Data, Reader, Xlsx};

fn read_xlsx_by_position(path: &str, col_indices: &[usize]) -> Vec<Vec<f64>> {
    let mut workbook: Xlsx<_> = open_workbook(path).unwrap();
    let sheet_name = workbook.sheet_names()[0].clone();
    let range = workbook.worksheet_range(&sheet_name).expect("Failed to read sheet");
    // ... positional column reading
}

#[test]
fn rsi_matches_python_reference() {
    let path = "resources/data/btc-rsi.xlsx";
    let cols = read_xlsx_by_position(path, &[1, 6]); // close, rsi
    // ... verify Rust RSI matches Python within 1e-4 tolerance
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn sma_always_in_range(data in prop::collection::vec(0.0..1000.0, 20..100)) {
        let (output, _, _, _) = sma_batch(&data, 14);

        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for val in output.iter().filter(|v| !v.is_nan()) {
            prop_assert!(*val >= min && *val <= max,
                "SMA {} outside data range [{}, {}]", val, min, max);
        }
    }

    #[test]
    fn rsi_always_0_to_100(data in prop::collection::vec(0.1..1000.0, 20..100)) {
        let (output, _) = rsi_batch(&data, 14);

        for val in output.iter().filter(|v| !v.is_nan()) {
            prop_assert!(*val >= 0.0 && *val <= 100.0,
                "RSI {} outside [0, 100]", val);
        }
    }
}
```

---

## Dependencies Required

```toml
# Cargo.toml additions

[dependencies]
# Existing...
thiserror = "2"
tokio = { version = "1", features = ["full", "sync"] }

# For TA module
circular-buffer = "0.1"    # Fixed-size ring buffer

[dev-dependencies]
approx = "0.5"             # Float comparison
proptest = "1"             # Property-based testing
criterion = "0.5"          # Benchmarking
```

---

## Summary: Keep vs. Change

| Python Pattern | Rust Decision | Rationale |
|----------------|---------------|-----------|
| `calc()` + `update()` | **Keep** | Proven, matches streaming + batch |
| Numba JIT functions | **Keep** (as Rust fns) | Already pure, stateless |
| Indicator composition | **Keep** | Owned fields instead of references |
| `_keys` string list | **Change** → Output struct | Type safety |
| `_period_config` dict | **Change** → Config struct | Compile-time validation |
| `_initialized` bool | **Change** → State enum | Explicit warmup handling |
| `_memory` optional | **Change** → Always store (ring buffer) | Simpler, bounded memory |
| `to_numpy(get=...)` | **Change** → `latest()` + `history()` | No string dispatch |
| Inheritance | **Change** → Trait | Rust idiom |

---

## Next Steps

1. Review this plan for feedback
2. Start Phase 1 (Foundation) implementation
3. Implement Phase 2 (Math Layer) with unit tests
4. Build Phase 3 indicators, test against Python reference
5. Integrate with Asterion pipeline

---

**Related Documents:**
- [asterion.md](asterion.md) — Pipeline architecture
- [BRAIN.md](BRAIN.md) — Engineering principles
- [rolling_ta/extras/numba.py](rolling_ta/extras/numba.py) — Math reference
