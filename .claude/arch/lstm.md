# LSTM State Vector Definition (Rust + tch)

This document defines the state vector used as input to an LSTM model implemented in Rust with the tch crate (PyTorch bindings). The design supports multi-step OHLCV forecasting and alignment with LR(3/7/14) "feeler" forecasts for PPO confidence scoring.

---

## 1. Definition of the State Vector

The LSTM input is a sliding window of the last T candles, where each candle is represented by a fixed-length feature vector.

```
X ∈ R^(T×F)
```

Where:
- **T** = lookback window (e.g., 64 candles)
- **F** = number of features per candle (here, 29)

With batching: `[batch_size, T, F]`

This is distinct from the LSTM's internal hidden and cell states.

---

## 2. Feature Set Per Candle (F = 29)

Each timestep contains normalized, stationary features derived from OHLCV and indicators.

### A. Core OHLCV (5)

| Index | Feature |
|-------|---------|
| 0 | `ret_close = log(C_t / C_{t-1})` |
| 1 | `hl_range = (High - Low) / Close` |
| 2 | `co_range = (Close - Open) / Close` |
| 3 | `ret_volume = log(V_t / V_{t-1})` |
| 4 | `close_z = rolling z-score of Close` |

### B. Trend Indicators (7)

| Index | Feature |
|-------|---------|
| 5 | `ema_fast_norm` |
| 6 | `ema_slow_norm` |
| 7 | `hma_norm` |
| 8 | `macd_line` |
| 9 | `macd_hist` |
| 10 | `adx` |
| 11 | `lr_slope_14` |

*(All normalized by Close or scaled.)*

### C. Momentum Indicators (4)

| Index | Feature |
|-------|---------|
| 12 | `rsi` |
| 13 | `stochrsi_k` |
| 14 | `roc` |
| 15 | `bop_sma` |

### D. Volatility Indicators (4)

| Index | Feature |
|-------|---------|
| 16 | `atr_norm = ATR / Close` |
| 17 | `bb_width` |
| 18 | `donchian_width` |
| 19 | `tr_norm` |

### E. Volume / Flow Indicators (3)

| Index | Feature |
|-------|---------|
| 20 | `obv_z` |
| 21 | `vwap_dist = (Close - VWAP) / VWAP` |
| 22 | `cmf_or_mfi` |

### F. LR Feeler Context (6)

| Index | Feature |
|-------|---------|
| 23 | `lr3_slope` |
| 24 | `lr7_slope` |
| 25 | `lr14_slope` |
| 26 | `lr3_forecast_delta` |
| 27 | `lr7_forecast_delta` |
| 28 | `lr14_forecast_delta` |

Where: `lrX_forecast_delta = (LRX_next - Close) / Close`

---

## 3. Fixed Feature Ordering

The state vector per timestep is ordered as:

```
 0  ret_close
 1  hl_range
 2  co_range
 3  ret_volume
 4  close_z

 5  ema_fast_norm
 6  ema_slow_norm
 7  hma_norm
 8  macd_line
 9  macd_hist
10  adx
11  lr_slope_14

12  rsi
13  stochrsi_k
14  roc
15  bop_sma

16  atr_norm
17  bb_width
18  donchian_width
19  tr_norm

20  obv_z
21  vwap_dist
22  cmf_or_mfi

23  lr3_slope
24  lr7_slope
25  lr14_slope
26  lr3_forecast_delta
27  lr7_forecast_delta
28  lr14_forecast_delta
```

**Do not change this order once trained.**

---

## 4. Tensor Shape Conventions (tch)

```
Input  X: [batch_size, T, F]
Output Y: [batch_size, H]
```

Where:
- **T** = 64 lookback window (typical)
- **F** = 29 features
- **H** = 5-20 forecast horizon (Close only)

Example:

```rust
let x = Tensor::zeros(&[batch_size, T, F], (Kind::Float, device));
```

---

## 5. Normalization Strategy

| Type | Strategy |
|------|----------|
| Returns and ranges | Already scale-free |
| Bounded indicators | Map to [-1, 1] |
| Continuous indicators | Rolling z-score |

**Bounded indicator scaling:**

```rust
rsi_scaled = (rsi - 50.0) / 50.0
adx_scaled = adx / 100.0
```

**Rolling z-score:**

```
z = (x - mean_N) / std_N
```

Maintain rolling statistics in streaming mode.

---

## 6. Rust Struct for One Timestep

```rust
#[derive(Clone, Copy)]
pub struct LstmState {
    // Core OHLCV
    pub ret_close: f32,
    pub hl_range: f32,
    pub co_range: f32,
    pub ret_volume: f32,
    pub close_z: f32,

    // Trend
    pub ema_fast_norm: f32,
    pub ema_slow_norm: f32,
    pub hma_norm: f32,
    pub macd_line: f32,
    pub macd_hist: f32,
    pub adx: f32,
    pub lr_slope_14: f32,

    // Momentum
    pub rsi: f32,
    pub stochrsi_k: f32,
    pub roc: f32,
    pub bop_sma: f32,

    // Volatility
    pub atr_norm: f32,
    pub bb_width: f32,
    pub donchian_width: f32,
    pub tr_norm: f32,

    // Volume / Flow
    pub obv_z: f32,
    pub vwap_dist: f32,
    pub cmf_or_mfi: f32,

    // LR Feelers
    pub lr3_slope: f32,
    pub lr7_slope: f32,
    pub lr14_slope: f32,
    pub lr3_forecast_delta: f32,
    pub lr7_forecast_delta: f32,
    pub lr14_forecast_delta: f32,
}
```

Maintain: `Vec<LstmState>` with length = T

---

## 7. Converting to a tch::Tensor

```rust
fn states_to_tensor(states: &[LstmState], device: Device) -> Tensor {
    let t = states.len();
    let f = 29;
    let mut data = Vec::with_capacity(t * f);

    for s in states {
        data.extend_from_slice(&[
            s.ret_close, s.hl_range, s.co_range, s.ret_volume, s.close_z,
            s.ema_fast_norm, s.ema_slow_norm, s.hma_norm,
            s.macd_line, s.macd_hist, s.adx, s.lr_slope_14,
            s.rsi, s.stochrsi_k, s.roc, s.bop_sma,
            s.atr_norm, s.bb_width, s.donchian_width, s.tr_norm,
            s.obv_z, s.vwap_dist, s.cmf_or_mfi,
            s.lr3_slope, s.lr7_slope, s.lr14_slope,
            s.lr3_forecast_delta, s.lr7_forecast_delta, s.lr14_forecast_delta,
        ]);
    }

    Tensor::of_slice(&data)
        .view([1, t as i64, f])
        .to_device(device)
}
```

---

## 8. Output Alignment

The LSTM predicts: `Y_hat ∈ [1, H]`

Representing future Close returns or deltas. Reconstruct price paths before computing R² against LR feelers.

---

## 9. PPO Integration

Expose to PPO:
- `r2_lr3`
- `r2_lr7`
- `r2_lr14`
- `r2_weighted`
- Optional: LSTM forecast slope

These belong to the PPO environment state, not the LSTM input.

---

## Summary

The LSTM state vector is a rolling window of **T × 29** normalized features per candle, combining:
- OHLCV-derived values
- Trend/momentum/volatility/volume indicators
- LR feeler context

Encoded as a `tch::Tensor` of shape `[batch, T, F]`.

This design supports:
1. Stable multi-step forecasts
2. Alignment with LR feelers
3. Downstream PPO confidence modeling
