# DCA Pipeline Integration Tests Plan

## Analysis

### Task Requirements
Write a single test file to validate:
1. Load 100,000 candles each for BTC, ETH, LTC from `[btc_usdt_ohlcv, eth_usdt_ohlcv, ltc_usdt_ohlcv]` tables
2. Aggregate into 5m and 15m timeframes using `PipelineDataDistributor`
3. Run TA calculations for state vector (with timing metrics)
4. Set TA output back in distributor for normalization stage

### Concerns Identified

1. **Database Dependency**: Tests require PostgreSQL with populated data
   - Solution: Mark as `#[ignore]` by default, run via `cargo test -- --ignored`

2. **Data Volume**: 300k total rows is significant
   - Solution: Use existing `OhlcvReader` batch streaming pattern

3. **Time Measurement**: Need accurate benchmarking
   - Solution: Use `std::time::Instant` for microsecond precision

4. **Async Runtime**: Integration tests need tokio
   - Solution: Use `#[tokio::test]` attribute

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Unit tests with mock data | Fast, no DB needed | Doesn't test real performance | Rejected |
| Separate test files per component | Isolation | Harder to test integration flow | Rejected |
| Single comprehensive test file | Tests full flow, matches requirements | Requires DB | **Selected** |

---

## Implementation Plan

### File Location
`src-tauri/src/pipeline/tests/dca_pipeline_integration.rs`

### Test Structure

```rust
// Module structure
mod dca_pipeline_integration {
    // 1. Database loading tests
    mod db_loading {
        fn test_load_100k_btc_candles()
        fn test_load_100k_eth_candles()
        fn test_load_100k_ltc_candles()
        fn test_load_all_assets_parallel()
    }

    // 2. Aggregation tests
    mod aggregation {
        fn test_aggregate_1m_to_5m()
        fn test_aggregate_1m_to_15m()
        fn test_batch_aggregation_100k_candles()
    }

    // 3. TA calculation tests with timing
    mod ta_calculation {
        fn test_ppo_dca_preset_100k_candles()
        fn test_individual_indicator_timing()
        fn test_multi_asset_ta_calculation()
    }

    // 4. State builder integration
    mod state_builder {
        fn test_state_builder_with_ta_output()
        fn test_42_dim_observation_generation()
    }
}
```

### Key Components to Use

| Component | Source | Purpose |
|-----------|--------|---------|
| `OhlcvReader` | `db::reader` | Load from PostgreSQL |
| `PipelineDataDistributor` | `pipeline::distributor` | Aggregate timeframes |
| `IndicatorFactory` | `pipeline::ta::factory` | Create indicators |
| `ppo_dca_preset()` | `pipeline::ta::presets` | PPO DCA config |
| `DcaStateBuilder` | `rl::dca::state_builder` | Build 42-dim state |
| `TAOutputValue` | `rl::dca::state_builder` | TA output enum |

### Database Setup Helper

```rust
async fn setup_db() -> PgPool {
    let db_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for integration tests");
    PgPool::connect(&db_url).await.expect("Failed to connect")
}
```

### Timing Metrics Pattern

```rust
fn measure<F, R>(name: &str, f: F) -> R
where F: FnOnce() -> R {
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    println!("[TIMING] {}: {:?}", name, elapsed);
    result
}
```

---

## Test Cases Detail

### 1. Database Loading Tests

**test_load_100k_candles_per_asset**
- Load 100k candles from each table using `OhlcvReader`
- Verify count matches expected
- Measure load time per asset
- Verify oldest/newest timestamps are sensible

### 2. Aggregation Tests

**test_distribute_100k_to_5m_and_15m**
- Create `PipelineDataDistributor::new(candles_1m)`
- Verify M1 count: 100,000
- Verify M5 count: ~20,000 (100k/5)
- Verify M15 count: ~6,666 (100k/15)
- Measure aggregation time

**test_ohlcv_aggregation_correctness**
- Verify OHLCV rules:
  - Open = first candle's open
  - High = max of all highs
  - Low = min of all lows
  - Close = last candle's close
  - Volume = sum of volumes

### 3. TA Calculation Tests

**test_ppo_dca_preset_all_indicators**
- Create indicators via `IndicatorFactory::create_all(&ppo_dca_preset())`
- Process all candles through each indicator
- Time per indicator:
  - SMA(50): expect < 50ms for 100k
  - SMA(200): expect < 50ms for 100k
  - RSI(14): expect < 100ms for 100k
  - Stochastic(21,3,3): expect < 150ms for 100k
  - ADX(14): expect < 200ms for 100k
  - BB(20,2): expect < 100ms for 100k
  - ATR(14): expect < 50ms for 100k
  - MFI(14): expect < 100ms for 100k
  - ROC(12): expect < 50ms for 100k
  - VWAP: expect < 100ms for 100k

**test_multi_asset_ta_parallel**
- Process BTC, ETH, LTC in parallel using tokio::spawn
- Measure total wall time vs sequential

### 4. State Builder Tests

**test_state_builder_assembly**
- Feed TA outputs to `DcaStateBuilder`
- Build `DcaObservation42`
- Validate all 42 fields within bounds
- Verify no NaN/infinite values

---

## Expected Output Format

```
[TEST] Loading BTC candles...
[TIMING] BTC load (100k): 245ms
[TEST] Loading ETH candles...
[TIMING] ETH load (100k): 238ms
[TEST] Loading LTC candles...
[TIMING] LTC load (100k): 251ms

[TEST] Aggregating BTC to 5m/15m...
[TIMING] BTC aggregation: 12ms
  - M1: 100,000 candles
  - M5: 20,000 candles
  - M15: 6,666 candles

[TEST] Running TA calculations on BTC M1...
[TIMING] SMA(50): 23ms
[TIMING] SMA(200): 24ms
[TIMING] RSI(14): 67ms
[TIMING] Stochastic(21,3,3): 89ms
[TIMING] ADX(14): 145ms
[TIMING] BB(20,2): 56ms
[TIMING] ATR(14): 34ms
[TIMING] MFI(14): 78ms
[TIMING] ROC(12): 21ms
[TIMING] VWAP: 45ms
[TIMING] All TA (BTC): 582ms

[TEST] Building state vectors...
[TIMING] State assembly (1000 samples): 12ms
[TEST] Observation validation: 1000/1000 valid
```

---

## Dependencies Required

```toml
# In src-tauri/Cargo.toml [dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

---

## Running Tests

```bash
# Set database URL
export DATABASE_URL="postgresql://user:pass@localhost/asterion"

# Run ignored integration tests
cargo test dca_pipeline_integration -- --ignored --nocapture

# Run specific test
cargo test test_load_100k_candles -- --ignored --nocapture
```
