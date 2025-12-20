# Pipeline Architecture Design

> **Status**: Draft
> **Created**: 2025-12-10
> **Pattern**: Observer + Chain of Responsibility

---

## Table of Contents

- [Pipeline Architecture Design](#pipeline-architecture-design)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Module Structure](#module-structure)
  - [Implementation Tasks](#implementation-tasks)
    - [Phase 1: Foundation](#phase-1-foundation)
    - [Phase 2: Stages](#phase-2-stages)
    - [Phase 3: Integration](#phase-3-integration)
    - [Phase 4: Kraken Client](#phase-4-kraken-client)
  - [1. Error Types](#1-error-types)
  - [2. Domain Types](#2-domain-types)
  - [3. Pipeline Trait](#3-pipeline-trait)
  - [4. Tap Registry](#4-tap-registry)
  - [5. Stage Implementations](#5-stage-implementations)
    - [5.1 TACalculator (Example)](#51-tacalculator-example)
    - [5.2 Normalizer (Welford's Algorithm)](#52-normalizer-welfords-algorithm)
  - [6. UIBridge](#6-uibridge)
    - [Frontend Event Listeners (TypeScript)](#frontend-event-listeners-typescript)
  - [Data Flow Diagram](#data-flow-diagram)
  - [Design Decisions](#design-decisions)
  - [Dependencies Required](#dependencies-required)
  - [Related Documents](#related-documents)

---

## Overview

The pipeline handles data flow from Kraken WebSocket through ML inference, with **tap points** at each stage for UI debugging:

```
WebSocket → [OHLCV] → [TA Calc] → [Normalize] → [ML] → [Action]
              ↓           ↓            ↓           ↓
             UI          UI           UI          UI
           (tap 1)     (tap 2)      (tap 3)     (tap 4)
```

**Core Pattern**: Each pipeline stage implements `PipelineStage<In, Out>` trait, which:
1. Processes input → output
2. Emits output to a `broadcast::Sender` (tap point)
3. Returns output for the next stage

---

## Module Structure

```
src-tauri/src/
├── lib.rs
├── pipeline/
│   ├── mod.rs              # Pipeline orchestrator + UIBridge
│   ├── error.rs            # thiserror definitions
│   ├── types.rs            # NewType domain types (no primitive obsession)
│   ├── traits.rs           # PipelineStage trait
│   ├── stages/
│   │   ├── mod.rs
│   │   ├── market.rs       # MarketActor (WS → OHLCV)
│   │   ├── technical.rs    # TACalculator (OHLCV → TABundle)
│   │   ├── normalizer.rs   # Normalizer (TABundle → NormState)
│   │   └── inference.rs    # MLActor (NormState → Action)
│   └── taps.rs             # Broadcast channel registry
└── kraken/
    ├── mod.rs
    ├── auth.rs             # HMAC-SHA512 signing
    ├── rest.rs             # REST client
    └── ws.rs               # WebSocket client
```

---

## Implementation Tasks

### Phase 1: Foundation
- [ ] Create `pipeline/error.rs` — PipelineError enum with thiserror
- [ ] Create `pipeline/types.rs` — NewType wrappers (Price, Volume, Timestamp)
- [ ] Create `pipeline/traits.rs` — PipelineStage trait definition
- [ ] Create `pipeline/taps.rs` — TapRegistry struct

### Phase 2: Stages
- [ ] Create `pipeline/stages/market.rs` — MarketActor (WS consumer)
- [ ] Create `pipeline/stages/technical.rs` — TACalculator (RSI, MACD, EMA)
- [ ] Create `pipeline/stages/normalizer.rs` — Welford online normalization
- [ ] Create `pipeline/stages/inference.rs` — ML inference stub

### Phase 3: Integration
- [ ] Create `pipeline/mod.rs` — UIBridge + orchestrator
- [ ] Wire TapRegistry into Tauri app state
- [ ] Add Tauri commands for pipeline control (start/stop/status)

### Phase 4: Kraken Client
- [ ] Create `kraken/auth.rs` — HMAC-SHA512 signing
- [ ] Create `kraken/rest.rs` — REST endpoints (get-ohlc-data, get-tradable-asset-pairs)
- [ ] Create `kraken/ws.rs` — WebSocket OHLCV subscription

---

## 1. Error Types

**File**: `src-tauri/src/pipeline/error.rs`

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("WebSocket connection failed: {0}")]
    WebSocketError(String),

    #[error("Channel send failed: {0}")]
    ChannelSend(String),

    #[error("Channel closed unexpectedly")]
    ChannelClosed,

    #[error("Normalization failed: {reason}")]
    NormalizationError { reason: String },

    #[error("ML inference failed: {0}")]
    InferenceError(String),

    #[error("Kraken API error: {code} - {message}")]
    KrakenApi { code: i32, message: String },

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Alias for consistency with Result<T> pattern
pub type PipelineResult<T> = Result<T, PipelineError>;
```

---

## 2. Domain Types

**File**: `src-tauri/src/pipeline/types.rs`

```rust
use serde::{Deserialize, Serialize};

// ============================================================
// NewType Wrappers (prevents primitive obsession)
// ============================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Price(pub f64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Volume(pub f64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Timestamp(pub i64);

/// Normalized value — always in [-1, 1] or z-scored
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NormalizedValue(pub f64);

// ============================================================
// OHLCV (Tap 1: Raw Market Data)
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ohlcv {
    pub timestamp: Timestamp,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Volume,
    pub vwap: Price,
    pub count: u32,
}

// ============================================================
// TABundle (Tap 2: Technical Analysis)
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TABundle {
    pub timestamp: Timestamp,
    pub rsi: Option<f64>,
    pub macd: Option<MacdValues>,
    pub ema_fast: Option<Price>,
    pub ema_slow: Option<Price>,
    pub atr: Option<f64>,
    /// Raw OHLCV passed through for UI tap
    pub source_ohlcv: Ohlcv,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacdValues {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

// ============================================================
// NormalizedState (Tap 3: What the Agent Sees)
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedState {
    pub timestamp: Timestamp,
    /// Fixed-size feature vector for ML input
    pub features: Vec<NormalizedValue>,
    /// Feature names for debugging UI
    pub feature_names: Vec<String>,
    /// Pass through for UI tap
    pub source_ta: TABundle,
}

// ============================================================
// MLOutput (Tap 4: Agent Decision)
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Hold,
    Buy { confidence: f64 },
    Sell { confidence: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLOutput {
    pub timestamp: Timestamp,
    pub action: Action,
    /// Critic head output (V(s) estimate)
    pub value_estimate: f64,
    /// Pass through for UI tap
    pub source_state: NormalizedState,
}
```

---

## 3. Pipeline Trait

**File**: `src-tauri/src/pipeline/traits.rs`

```rust
use tokio::sync::broadcast;
use crate::pipeline::error::PipelineResult;

/// Each pipeline stage:
/// 1. Processes input → output
/// 2. Emits output to broadcast channel (tap point)
/// 3. Returns output for next stage
///
/// # Contract
/// Implementors MUST call `self.tap().send(output.clone())` within `process()`
pub trait PipelineStage<In, Out>: Send + Sync
where
    In: Clone + Send + 'static,
    Out: Clone + Send + 'static,
{
    /// Access the broadcast sender for this stage's output
    fn tap(&self) -> &broadcast::Sender<Out>;

    /// Process input, emit to tap, return output
    fn process(&mut self, input: In) -> PipelineResult<Out>;

    /// Human-readable stage name for logging
    fn name(&self) -> &'static str;
}
```

---

## 4. Tap Registry

**File**: `src-tauri/src/pipeline/taps.rs`

```rust
use tokio::sync::broadcast;
use crate::pipeline::types::{Ohlcv, TABundle, NormalizedState, MLOutput};

const CHANNEL_CAPACITY: usize = 64;

/// Central registry of all pipeline tap points.
/// UIBridge subscribes to these for frontend emission.
pub struct TapRegistry {
    pub ohlcv: broadcast::Sender<Ohlcv>,
    pub ta_bundle: broadcast::Sender<TABundle>,
    pub normalized: broadcast::Sender<NormalizedState>,
    pub ml_output: broadcast::Sender<MLOutput>,
}

impl TapRegistry {
    pub fn new() -> Self {
        Self {
            ohlcv: broadcast::channel(CHANNEL_CAPACITY).0,
            ta_bundle: broadcast::channel(CHANNEL_CAPACITY).0,
            normalized: broadcast::channel(CHANNEL_CAPACITY).0,
            ml_output: broadcast::channel(CHANNEL_CAPACITY).0,
        }
    }

    /// Create receivers for all taps (used by UIBridge)
    pub fn subscribe_all(
        &self,
    ) -> (
        broadcast::Receiver<Ohlcv>,
        broadcast::Receiver<TABundle>,
        broadcast::Receiver<NormalizedState>,
        broadcast::Receiver<MLOutput>,
    ) {
        (
            self.ohlcv.subscribe(),
            self.ta_bundle.subscribe(),
            self.normalized.subscribe(),
            self.ml_output.subscribe(),
        )
    }
}

impl Default for TapRegistry {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## 5. Stage Implementations

### 5.1 TACalculator (Example)

**File**: `src-tauri/src/pipeline/stages/technical.rs`

```rust
use tokio::sync::broadcast;
use std::collections::VecDeque;
use crate::pipeline::{
    error::{PipelineError, PipelineResult},
    traits::PipelineStage,
    types::{Ohlcv, TABundle, MacdValues, Price},
};

const RSI_PERIOD: usize = 14;
const EMA_FAST_PERIOD: usize = 12;
const EMA_SLOW_PERIOD: usize = 26;
const MACD_SIGNAL_PERIOD: usize = 9;

pub struct TACalculator {
    tap_sender: broadcast::Sender<TABundle>,
    // Rolling windows for O(1) updates
    close_history: VecDeque<f64>,
    gain_history: VecDeque<f64>,
    loss_history: VecDeque<f64>,
    // EMA state
    ema_fast: Option<f64>,
    ema_slow: Option<f64>,
    macd_signal: Option<f64>,
    tick_count: usize,
}

impl TACalculator {
    pub fn new(tap_sender: broadcast::Sender<TABundle>) -> Self {
        Self {
            tap_sender,
            close_history: VecDeque::with_capacity(EMA_SLOW_PERIOD + 1),
            gain_history: VecDeque::with_capacity(RSI_PERIOD),
            loss_history: VecDeque::with_capacity(RSI_PERIOD),
            ema_fast: None,
            ema_slow: None,
            macd_signal: None,
            tick_count: 0,
        }
    }

    fn calculate_rsi(&self) -> Option<f64> {
        if self.gain_history.len() < RSI_PERIOD {
            return None;
        }
        let avg_gain: f64 = self.gain_history.iter().sum::<f64>() / RSI_PERIOD as f64;
        let avg_loss: f64 = self.loss_history.iter().sum::<f64>() / RSI_PERIOD as f64;

        if avg_loss == 0.0 {
            return Some(100.0);
        }
        let rs = avg_gain / avg_loss;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }

    fn update_ema(prev: Option<f64>, price: f64, period: usize, tick: usize) -> Option<f64> {
        if tick < period {
            return None;
        }
        let multiplier = 2.0 / (period as f64 + 1.0);
        match prev {
            Some(prev_ema) => Some((price - prev_ema) * multiplier + prev_ema),
            None => Some(price), // First EMA = price (SMA approximation)
        }
    }

    fn calculate_macd(&self) -> Option<MacdValues> {
        match (self.ema_fast, self.ema_slow, self.macd_signal) {
            (Some(fast), Some(slow), Some(signal)) => {
                let macd_line = fast - slow;
                Some(MacdValues {
                    macd_line,
                    signal_line: signal,
                    histogram: macd_line - signal,
                })
            }
            _ => None,
        }
    }
}

impl PipelineStage<Ohlcv, TABundle> for TACalculator {
    fn tap(&self) -> &broadcast::Sender<TABundle> {
        &self.tap_sender
    }

    fn name(&self) -> &'static str {
        "TACalculator"
    }

    fn process(&mut self, input: Ohlcv) -> PipelineResult<TABundle> {
        let close = input.close.0;
        self.tick_count += 1;

        // Update RSI rolling windows
        if let Some(&prev_close) = self.close_history.back() {
            let change = close - prev_close;
            if change >= 0.0 {
                self.gain_history.push_back(change);
                self.loss_history.push_back(0.0);
            } else {
                self.gain_history.push_back(0.0);
                self.loss_history.push_back(change.abs());
            }
            if self.gain_history.len() > RSI_PERIOD {
                self.gain_history.pop_front();
                self.loss_history.pop_front();
            }
        }

        // Update close history
        self.close_history.push_back(close);
        if self.close_history.len() > EMA_SLOW_PERIOD + 1 {
            self.close_history.pop_front();
        }

        // Update EMAs
        self.ema_fast = Self::update_ema(self.ema_fast, close, EMA_FAST_PERIOD, self.tick_count);
        self.ema_slow = Self::update_ema(self.ema_slow, close, EMA_SLOW_PERIOD, self.tick_count);

        // Update MACD signal line
        if let (Some(fast), Some(slow)) = (self.ema_fast, self.ema_slow) {
            let macd_line = fast - slow;
            self.macd_signal = Self::update_ema(
                self.macd_signal,
                macd_line,
                MACD_SIGNAL_PERIOD,
                self.tick_count.saturating_sub(EMA_SLOW_PERIOD),
            );
        }

        let bundle = TABundle {
            timestamp: input.timestamp,
            rsi: self.calculate_rsi(),
            macd: self.calculate_macd(),
            ema_fast: self.ema_fast.map(Price),
            ema_slow: self.ema_slow.map(Price),
            atr: None, // TODO: implement ATR
            source_ohlcv: input,
        };

        // Emit to tap — ignore receiver count (lagging receivers drop old messages)
        let _ = self.tap_sender.send(bundle.clone());

        Ok(bundle)
    }
}
```

### 5.2 Normalizer (Welford's Algorithm)

**File**: `src-tauri/src/pipeline/stages/normalizer.rs`

```rust
use tokio::sync::broadcast;
use crate::pipeline::{
    error::PipelineResult,
    traits::PipelineStage,
    types::{TABundle, NormalizedState, NormalizedValue, Timestamp},
};

/// Online normalizer using Welford's algorithm for running mean/variance
pub struct Normalizer {
    tap_sender: broadcast::Sender<NormalizedState>,
    feature_stats: Vec<WelfordState>,
    feature_names: Vec<String>,
    epsilon: f64,
}

struct WelfordState {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
}

impl WelfordState {
    fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 1.0; // Avoid division by zero
        }
        self.m2 / (self.count - 1) as f64
    }

    fn normalize(&self, value: f64, epsilon: f64) -> f64 {
        (value - self.mean) / (self.variance().sqrt() + epsilon)
    }
}

impl Normalizer {
    pub fn new(tap_sender: broadcast::Sender<NormalizedState>) -> Self {
        let feature_names = vec![
            "close".into(),
            "volume".into(),
            "rsi".into(),
            "macd_histogram".into(),
            "ema_diff".into(),
        ];
        let feature_stats = (0..feature_names.len())
            .map(|_| WelfordState::new())
            .collect();

        Self {
            tap_sender,
            feature_stats,
            feature_names,
            epsilon: 1e-8,
        }
    }

    fn extract_features(&self, ta: &TABundle) -> Vec<f64> {
        vec![
            ta.source_ohlcv.close.0,
            ta.source_ohlcv.volume.0,
            ta.rsi.unwrap_or(50.0), // Default to neutral RSI
            ta.macd.as_ref().map(|m| m.histogram).unwrap_or(0.0),
            match (ta.ema_fast, ta.ema_slow) {
                (Some(fast), Some(slow)) => fast.0 - slow.0,
                _ => 0.0,
            },
        ]
    }
}

impl PipelineStage<TABundle, NormalizedState> for Normalizer {
    fn tap(&self) -> &broadcast::Sender<NormalizedState> {
        &self.tap_sender
    }

    fn name(&self) -> &'static str {
        "Normalizer"
    }

    fn process(&mut self, input: TABundle) -> PipelineResult<NormalizedState> {
        let raw_features = self.extract_features(&input);

        // Update stats and normalize
        let normalized: Vec<NormalizedValue> = raw_features
            .iter()
            .zip(self.feature_stats.iter_mut())
            .map(|(&value, stats)| {
                stats.update(value);
                NormalizedValue(stats.normalize(value, self.epsilon))
            })
            .collect();

        let state = NormalizedState {
            timestamp: input.timestamp,
            features: normalized,
            feature_names: self.feature_names.clone(),
            source_ta: input,
        };

        let _ = self.tap_sender.send(state.clone());

        Ok(state)
    }
}
```

---

## 6. UIBridge

**File**: `src-tauri/src/pipeline/mod.rs`

```rust
pub mod error;
pub mod stages;
pub mod taps;
pub mod traits;
pub mod types;

use tauri::{AppHandle, Emitter};
use tokio::sync::broadcast;
use crate::pipeline::types::{Ohlcv, TABundle, NormalizedState, MLOutput};

pub struct UIBridge {
    app: AppHandle,
}

impl UIBridge {
    pub fn new(app: AppHandle) -> Self {
        Self { app }
    }

    /// Spawn listener tasks for all taps → emit to frontend
    pub fn start(
        self,
        mut ohlcv_rx: broadcast::Receiver<Ohlcv>,
        mut ta_rx: broadcast::Receiver<TABundle>,
        mut norm_rx: broadcast::Receiver<NormalizedState>,
        mut ml_rx: broadcast::Receiver<MLOutput>,
    ) {
        let app = self.app.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Ok(ohlcv) = ohlcv_rx.recv() => {
                        let _ = app.emit("pipeline:ohlcv", &ohlcv);
                    }
                    Ok(ta) = ta_rx.recv() => {
                        let _ = app.emit("pipeline:ta", &ta);
                    }
                    Ok(norm) = norm_rx.recv() => {
                        let _ = app.emit("pipeline:normalized", &norm);
                    }
                    Ok(ml) = ml_rx.recv() => {
                        let _ = app.emit("pipeline:action", &ml);
                    }
                }
            }
        });
    }
}
```

### Frontend Event Listeners (TypeScript)

```typescript
// src/hooks/usePipelineTaps.ts
import { listen } from '@tauri-apps/api/event';
import { useEffect } from 'react';
import { useStore } from '@/stores/pipeline-store';

export function usePipelineTaps() {
  const { setOhlcv, setTA, setNormalized, setAction } = useStore();

  useEffect(() => {
    const unlisteners = Promise.all([
      listen<Ohlcv>('pipeline:ohlcv', (e) => setOhlcv(e.payload)),
      listen<TABundle>('pipeline:ta', (e) => setTA(e.payload)),
      listen<NormalizedState>('pipeline:normalized', (e) => setNormalized(e.payload)),
      listen<MLOutput>('pipeline:action', (e) => setAction(e.payload)),
    ]);

    return () => {
      unlisteners.then((fns) => fns.forEach((fn) => fn()));
    };
  }, []);
}
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Pipeline Thread (tokio)                       │
│                                                                      │
│  WS ──► MarketActor ──► TACalculator ──► Normalizer ──► MLActor     │
│              │               │               │              │        │
│              ▼               ▼               ▼              ▼        │
│         tap.send()      tap.send()      tap.send()     tap.send()   │
└──────────────────────────────────────────────────────────────────────┘
                │               │               │              │
                └───────────────┴───────────────┴──────────────┘
                                        │
                                   TapRegistry
                                        │
                              ┌─────────┴─────────┐
                              │     UIBridge      │
                              │  app.emit(...)    │
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   React Frontend  │
                              │  listen("pipeline:*") │
                              └───────────────────┘
```

---

## Design Decisions

| Decision | Rationale | Reference |
|----------|-----------|-----------|
| **NewType wrappers** | Prevents `Price` confusion with `Volume` | `.agent/rules/rust-best-practice.md` |
| **`broadcast::Sender`** | Observer pattern; multiple receivers without blocking | `.agent/architecture/design-patterns.md` |
| **`source_*` fields** | Each stage carries its input for debugging taps | User requirement |
| **`PipelineResult<T>`** | Consistent with Result<T> pattern; no `.unwrap()` | `.agent/architecture/result-type.md` |
| **`VecDeque` windows** | O(1) rolling indicator updates | `plan.md` Phase 3 |
| **Welford's algorithm** | Online normalization without storing history | `plan.md` Phase 3 |
| **`thiserror`** | Idiomatic Rust error handling | `.agent/frameworks/rust.md` |

---

## Dependencies Required

Add to `src-tauri/Cargo.toml`:

```toml
[dependencies]
# Existing...
tokio = { version = "1", features = ["full", "sync"] }
thiserror = "2"

# New for pipeline
tokio-tungstenite = { version = "0.21", features = ["rustls-tls-webpki-roots"] }
futures-util = "0.3"
hmac = "0.12"
sha2 = "0.10"
base64 = "0.22"
```

---

## Related Documents

- [plan.md](../plan.md) — Original implementation plan
- [endpoints.md](./endpoints.md) — Kraken API reference
- [.agent/architecture/design-patterns.md](../../.agent/architecture/design-patterns.md) — Pattern catalog
- [.agent/frameworks/rust.md](../../.agent/frameworks/rust.md) — Rust best practices
