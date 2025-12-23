# PPO Actor-Critic Design for Multi-Asset Crypto Trading

Rust + tch, LSTM-augmented state

---

## 1. Design Goals

Your setup implies the PPO policy should:

- Learn from large-scale minute data (non-i.i.d., regime shifts)
- Exploit:
  - Raw market microstructure (OHLCV + indicators)
  - LSTM multi-step forecasts (5-20 candles)
  - Agreement metrics vs LR feelers (R²)
- Make decisions under:
  - Trend vs chop
  - Volatility regimes
  - Forecast confidence

Thus, the policy should:

1. Be recurrent (partial observability)
2. Separate representation learning from action/value heads
3. Treat LSTM outputs as exogenous latent signals, not replace PPO recurrence

---

## 2. High-Level Architecture

Recurrent PPO with shared encoder + GRU/LSTM core and dual heads:

```
State Vector s_t
      ↓
Feature Encoder (MLP)
      ↓
Recurrent Core (GRU or LSTM)
      ↓
Shared Latent h_t
      ├── Actor Head  → π(a_t | s_t)
      └── Critic Head → V(s_t)
```

And in parallel:

```
Market History → LSTM Forecaster → Forecast + R² signals
                              ↘
                               appended into PPO state
```

---

## 3. PPO Network Specification

### 3.1 Encoder

- Input dim = S (state vector size, defined below)
- MLP:
  - `Linear(S → 256) + ReLU`
  - `Linear(256 → 128) + ReLU`

### 3.2 Recurrent Core

Use GRU for stability and speed at scale:

| Parameter | Value |
|-----------|-------|
| Input | 128 |
| Hidden | 128 |
| Layers | 1 |

*(You can swap to LSTM if you want symmetry, but GRU is often enough.)*

### 3.3 Actor Head

```
Linear(128 → 64) + ReLU
Linear(64 → A)
```

Where A is action dimension:
- Example: `{short, flat, long}` → A = 3
- Or continuous position ∈ `[-1, 1]`

Use:
- **Softmax** for discrete
- **Gaussian (μ, σ)** for continuous

### 3.4 Critic Head

```
Linear(128 → 64) + ReLU
Linear(64 → 1)
```

Outputs scalar value.

### 3.5 Why This Works

- Shared encoder learns market structure
- Recurrent core captures position history, regime persistence, drawdown memory
- Separate heads stabilize PPO updates
- Scales well to millions of steps

---

## 4. PPO State Vector Definition

The PPO state is not raw OHLCV windows. It is a compressed, information-rich snapshot per timestep, including:

- Current market features (from your indicator stack)
- Outputs from LSTM forecaster
- LR agreement metrics
- Portfolio context
- Asset identity

### 4.1 Market Features (per asset) — ~20 dims

Reuse a reduced subset of your LSTM features:

**Price & microstructure (5):**

| Feature |
|---------|
| `ret_close` |
| `hl_range` |
| `co_range` |
| `ret_volume` |
| `close_z` |

**Trend (5):**

| Feature |
|---------|
| `ema_fast_norm` |
| `ema_slow_norm` |
| `macd_hist` |
| `adx` |
| `lr_slope_14` |

**Momentum (4):**

| Feature |
|---------|
| `rsi_scaled` |
| `stochrsi_k` |
| `roc` |
| `bop_sma` |

**Volatility (3):**

| Feature |
|---------|
| `atr_norm` |
| `bb_width` |
| `donchian_width` |

**Volume/flow (3):**

| Feature |
|---------|
| `obv_z` |
| `vwap_dist` |
| `cmf_or_mfi` |

→ **Total ≈ 20 features**

These are current candle only, not windows (recurrence handles memory).

### 4.2 LSTM Forecast Features — ~10 dims

From your LSTM forecaster (Close only), for horizon H (e.g., 10):

| Feature | Description |
|---------|-------------|
| `lstm_ret_1` | 1-step return |
| `lstm_ret_3` | 3-step return |
| `lstm_ret_5` | 5-step return |
| `lstm_ret_10` | 10-step return |
| `lstm_mean_ret` | Mean forecast return |
| `lstm_slope` | Forecast trend slope |
| `lstm_max_up` | Max upside in window |
| `lstm_max_down` | Max downside in window |
| `lstm_volatility` | Forecast volatility |
| `lstm_end_ret` | End-of-horizon return |

These summarize the forecast path shape.

### 4.3 LR Agreement Metrics — 4 dims

Your core confidence signals:

| Feature | Description |
|---------|-------------|
| `r2_lr3` | R² vs LR(3) |
| `r2_lr7` | R² vs LR(7) |
| `r2_lr14` | R² vs LR(14) |
| `r2_weighted` | Weighted average |

All scaled to `[-1, 1]`.

### 4.4 Portfolio / Agent State — 6 dims

Critical for PPO stability:

| Feature | Range/Description |
|---------|-------------------|
| `position` | ∈ `[-1, 1]` |
| `entry_ret` | Unrealized PnL |
| `equity_ret` | Since episode start |
| `drawdown` | Current drawdown |
| `exposure_time` | Scaled |
| `last_action` | One-hot or scalar |

### 4.5 Asset Identity — 3 dims

Since you train across BTC, ETH, LTC:

One-hot: `[is_btc, is_eth, is_ltc]`

Or embedding index.

### Final PPO State Size

| Component | Dims |
|-----------|------|
| Market features | ~20 |
| LSTM forecast summary | ~10 |
| LR agreement | 4 |
| Portfolio state | 6 |
| Asset id | 3 |
| **Total** | **~43 dimensions** |

Round to S = 48 if you want padding.

---

## 5. PPO State Tensor Shape

For recurrent PPO in tch:

```
[state_t] → [batch_size, S]
```

Hidden state:

```
h_t → [num_layers, batch_size, 128]
```

Sequence training (for BPTT):

```
[batch_size, T_seq, S]
```

Where `T_seq` ~ 32-128 steps.

---

## 6. Rust Struct for PPO State

```rust
pub struct PpoState {
    // Market (20)
    pub ret_close: f32,
    pub hl_range: f32,
    pub co_range: f32,
    pub ret_volume: f32,
    pub close_z: f32,

    pub ema_fast_norm: f32,
    pub ema_slow_norm: f32,
    pub macd_hist: f32,
    pub adx: f32,
    pub lr_slope_14: f32,

    pub rsi: f32,
    pub stochrsi_k: f32,
    pub roc: f32,
    pub bop_sma: f32,

    pub atr_norm: f32,
    pub bb_width: f32,
    pub donchian_width: f32,

    pub obv_z: f32,
    pub vwap_dist: f32,
    pub cmf_or_mfi: f32,

    // LSTM summary (10)
    pub lstm_ret_1: f32,
    pub lstm_ret_3: f32,
    pub lstm_ret_5: f32,
    pub lstm_ret_10: f32,
    pub lstm_mean_ret: f32,
    pub lstm_slope: f32,
    pub lstm_max_up: f32,
    pub lstm_max_down: f32,
    pub lstm_volatility: f32,
    pub lstm_end_ret: f32,

    // LR agreement (4)
    pub r2_lr3: f32,
    pub r2_lr7: f32,
    pub r2_lr14: f32,
    pub r2_weighted: f32,

    // Portfolio (6)
    pub position: f32,
    pub entry_ret: f32,
    pub equity_ret: f32,
    pub drawdown: f32,
    pub exposure_time: f32,
    pub last_action: f32,

    // Asset id (3)
    pub is_btc: f32,
    pub is_eth: f32,
    pub is_ltc: f32,
}
```

Flatten into `[S]` before tensor creation.

---

## 7. How LSTM Is Used by PPO

Your LSTM acts as a frozen or slowly updated world model.

### Option A — Pretrain LSTM, then freeze (Recommended)

- Train on all 12.6M candles
- Use outputs as deterministic features for PPO
- Most stable

### Option B — Periodic co-training

Alternate:
1. Update LSTM on latest buffer
2. Resume PPO

More adaptive, more complex.

### Option C — End-to-end (Not recommended initially)

- PPO backprop through LSTM
- Very unstable at scale

**Recommendation: Start with Option A.**

---

## 8. Training Regime

Given scale: **12.6M+ environment steps**

Use:
- Parallel envs per asset
- Rollout length: 128-256 steps
- Minibatches: 4-8
- PPO epochs: 3-5

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| γ (gamma) | 0.99-0.999 |
| λ (lambda) | 0.95 |
| clip | 0.1-0.2 |
| entropy coef | 0.001-0.01 |
| value coef | 0.5 |

Normalize advantages and state features.

---

## 9. Why This Setup Fits Your Problem

This architecture:

- Handles partial observability via recurrence
- Leverages:
  - Your indicator engineering
  - LSTM multi-step foresight
  - LR agreement confidence
- Scales to millions of timesteps
- Keeps PPO learning policy + risk, not price prediction

It lets the LSTM answer:
> "Where might price go?"

And PPO answer:
> "Given that, what is the optimal action under risk and regime?"

---

## Summary

Use a **recurrent PPO Actor-Critic** with:
- Shared MLP encoder
- GRU core (128 hidden)
- Dual heads

Operating on a **~43-48 dimensional state vector** that fuses:
- Current indicators
- LSTM forecast summaries
- LR agreement R² metrics
- Portfolio context
- Asset identity

This gives you:
1. Regime-aware decision making
2. Forecast-conditioned actions
3. A scalable path to learn from 12.6M minute candles
