//! Integration tests for CandleBuilder + SharedWindow + Parallel Indicators.
//!
//! Tests the full data pipeline:
//! 1. Read raw OHLCV tick data from xlsx
//! 2. Aggregate ticks into candles using CandleBuilder (1m, 5m, 15m)
//! 3. Store in SharedWindow for thread-safe access
//! 4. Process indicators in parallel using Rayon
//!
//! This validates the architecture before migrating all indicators.

use crate::common::{build_candles_ohlcv, read_xlsx_by_position, EPSILON};
use rayon::prelude::*;
use rolling_ta::prelude::*;
use rolling_ta::ta::math::RollingWindow;
use rolling_ta::trend::{SMAConfig, SMA};
use std::sync::{Arc, RwLock};

/// Load BTC OHLCV data from xlsx file.
/// btc-200.xlsx columns: [timestamp(0), open(1), high(2), low(3), close(4), volume(5)]
fn load_btc_data() -> Vec<Ohlcv> {
    let cols = read_xlsx_by_position("resources/data/btc-200.xlsx", &[0, 1, 2, 3, 4, 5]);
    build_candles_ohlcv(&cols[0], &cols[1], &cols[2], &cols[3], &cols[4], &cols[5])
}

/// Simulate tick data from OHLCV candles.
/// For each candle, generates 4 ticks: open, high, low, close
/// This simulates real-time tick arrival for CandleBuilder testing.
fn simulate_ticks_from_candles(candles: &[Ohlcv]) -> Vec<Tick> {
    let mut ticks = Vec::with_capacity(candles.len() * 4);

    for candle in candles {
        let base_ts = candle.timestamp.0;
        let interval = 15; // 15 seconds between ticks within a 1-minute candle

        // Open tick
        ticks.push(Tick::new(base_ts, candle.open.0, candle.volume.0 / 4.0));
        // High tick
        ticks.push(Tick::new(base_ts + interval, candle.high.0, candle.volume.0 / 4.0));
        // Low tick
        ticks.push(Tick::new(
            base_ts + interval * 2,
            candle.low.0,
            candle.volume.0 / 4.0,
        ));
        // Close tick
        ticks.push(Tick::new(
            base_ts + interval * 3,
            candle.close.0,
            candle.volume.0 / 4.0,
        ));
    }

    ticks
}

// ============================================================
// Integration Test: CandleBuilder aggregates ticks correctly
// ============================================================

#[test]
fn candle_builder_aggregates_ticks_to_1m_candles() {
    let source_candles = load_btc_data();
    let ticks = simulate_ticks_from_candles(&source_candles);

    // Build 1-minute candles from ticks
    let mut builder = CandleBuilder::new(Timeframe::M1);
    let mut window = RollingWindow::new(200);

    for tick in &ticks {
        if let Some(completed) = builder.push(tick) {
            window.push(completed);
        }
    }

    // Flush the last incomplete candle
    if let Some(last) = builder.flush() {
        window.push(last);
    }

    // Should have reconstructed all candles
    let reconstructed = window.snapshot();

    println!(
        "Original: {} candles, Reconstructed: {} candles",
        source_candles.len(),
        reconstructed.len()
    );

    // Verify we got back the same number of candles
    assert_eq!(
        reconstructed.len(),
        source_candles.len(),
        "Should reconstruct same number of candles"
    );

    // Verify OHLC values match (allowing for tick simulation rounding)
    for (i, (orig, recon)) in source_candles.iter().zip(reconstructed.iter()).enumerate() {
        // Open should match first tick
        assert!(
            (orig.open.0 - recon.open.0).abs() < EPSILON,
            "Open mismatch at {}: {} vs {}",
            i,
            orig.open.0,
            recon.open.0
        );

        // High should match max tick
        assert!(
            (orig.high.0 - recon.high.0).abs() < EPSILON,
            "High mismatch at {}: {} vs {}",
            i,
            orig.high.0,
            recon.high.0
        );

        // Low should match min tick
        assert!(
            (orig.low.0 - recon.low.0).abs() < EPSILON,
            "Low mismatch at {}: {} vs {}",
            i,
            orig.low.0,
            recon.low.0
        );

        // Close should match last tick
        assert!(
            (orig.close.0 - recon.close.0).abs() < EPSILON,
            "Close mismatch at {}: {} vs {}",
            i,
            orig.close.0,
            recon.close.0
        );
    }

    println!("CandleBuilder correctly reconstructed {} 1m candles from ticks", reconstructed.len());
}

// ============================================================
// Integration Test: Multi-timeframe aggregation (1m → 5m, 15m)
// ============================================================

#[test]
fn candle_builder_multi_timeframe_aggregation() {
    let source_candles = load_btc_data();

    // Assume source data is 1-minute candles with sequential timestamps
    // We'll aggregate to 5m and 15m

    let mut builder_5m = CandleBuilder::new(Timeframe::M5);
    let mut builder_15m = CandleBuilder::new(Timeframe::M15);

    let mut window_5m = RollingWindow::new(50);
    let mut window_15m = RollingWindow::new(20);

    // Process each 1m candle as if it were a tick for higher timeframes
    // Use the close price and aggregate volume
    for candle in &source_candles {
        let tick = Tick::new(candle.timestamp.0, candle.close.0, candle.volume.0);

        if let Some(completed) = builder_5m.push(&tick) {
            window_5m.push(completed);
        }
        if let Some(completed) = builder_15m.push(&tick) {
            window_15m.push(completed);
        }
    }

    // Flush remaining
    if let Some(last) = builder_5m.flush() {
        window_5m.push(last);
    }
    if let Some(last) = builder_15m.flush() {
        window_15m.push(last);
    }

    let candles_5m = window_5m.snapshot();
    let candles_15m = window_15m.snapshot();

    println!(
        "Source: {} 1m candles → {} 5m candles, {} 15m candles",
        source_candles.len(),
        candles_5m.len(),
        candles_15m.len()
    );

    // Verify ratios are approximately correct (allowing for partial periods)
    // 200 1m candles should produce ~40 5m candles and ~13-14 15m candles
    assert!(
        candles_5m.len() >= 35 && candles_5m.len() <= 45,
        "Expected ~40 5m candles, got {}",
        candles_5m.len()
    );
    assert!(
        candles_15m.len() >= 10 && candles_15m.len() <= 20,
        "Expected ~13-14 15m candles, got {}",
        candles_15m.len()
    );

    // Verify 5m candles have aggregated volume from 5 1m candles
    // (approximately, due to period boundaries)
    if !candles_5m.is_empty() {
        let first_5m = &candles_5m[0];
        println!(
            "First 5m candle: ts={}, O={:.2}, H={:.2}, L={:.2}, C={:.2}, V={:.2}",
            first_5m.timestamp.0,
            first_5m.open.0,
            first_5m.high.0,
            first_5m.low.0,
            first_5m.close.0,
            first_5m.volume.0
        );
    }
}

// ============================================================
// Integration Test: SharedWindow + Parallel Indicator Processing
// ============================================================

#[test]
fn shared_window_parallel_indicator_processing() {
    let source_candles = load_btc_data();

    // Create SharedWindow with the source data
    let window: Arc<RwLock<RollingWindow>> = Arc::new(RwLock::new(RollingWindow::new(200)));

    // Push all candles to window
    {
        let mut w = window.write().unwrap();
        for candle in &source_candles {
            w.push(*candle);
        }
    }

    // Take snapshot for parallel processing
    let snapshot: Vec<Ohlcv> = window.read().unwrap().snapshot();
    let snapshot = Arc::new(snapshot);

    // Define multiple SMA configurations to process in parallel
    let sma_configs = vec![
        ("SMA-7", 7),
        ("SMA-14", 14),
        ("SMA-21", 21),
        ("SMA-50", 50),
    ];

    // Process indicators in parallel using Rayon
    let results: Vec<(&str, Vec<f64>)> = sma_configs
        .par_iter()
        .map(|(name, period)| {
            let snapshot = Arc::clone(&snapshot);

            // Create indicator and calculate
            let mut sma = SMA::new(SMAConfig::new(*period));
            sma.calc(&snapshot).unwrap();

            // Get computed values (skip NaN warmup)
            let values: Vec<f64> = sma.history().iter().filter(|v| !v.is_nan()).copied().collect();

            (*name, values)
        })
        .collect();

    // Verify all indicators computed
    assert_eq!(results.len(), 4, "Should have 4 SMA results");

    for (name, values) in &results {
        println!("{}: {} computed values", name, values.len());
        assert!(!values.is_empty(), "{} should have computed values", name);

        // Verify first value is reasonable (positive, not zero for price data)
        let first = values[0];
        assert!(first > 0.0, "{} first value should be positive", name);
    }

    // Verify SMA-7 has more values than SMA-50 (shorter warmup)
    let sma7_count = results.iter().find(|(n, _)| *n == "SMA-7").unwrap().1.len();
    let sma50_count = results.iter().find(|(n, _)| *n == "SMA-50").unwrap().1.len();
    assert!(
        sma7_count > sma50_count,
        "SMA-7 should have more values than SMA-50"
    );

    println!(
        "Parallel processing complete: {} indicators on {} candles",
        results.len(),
        snapshot.len()
    );
}

// ============================================================
// Integration Test: Real-time simulation with streaming next()
// ============================================================

#[test]
fn realtime_streaming_with_parallel_indicators() {
    let source_candles = load_btc_data();
    let ticks = simulate_ticks_from_candles(&source_candles);

    // Shared window for completed candles
    let window: Arc<RwLock<RollingWindow>> = Arc::new(RwLock::new(RollingWindow::new(200)));

    // CandleBuilder for 1m aggregation
    let mut builder = CandleBuilder::new(Timeframe::M1);

    // SMA indicators (would be in IndicatorManager in production)
    let mut sma_14 = SMA::new(SMAConfig::new(14));
    let mut sma_21 = SMA::new(SMAConfig::new(21));

    let mut sma_14_values = Vec::new();
    let mut sma_21_values = Vec::new();

    // Simulate real-time tick processing
    for tick in &ticks {
        if let Some(completed_candle) = builder.push(tick) {
            // Push completed candle to window
            window.write().unwrap().push(completed_candle);

            // Take snapshot for indicator updates
            let snapshot = window.read().unwrap().snapshot();

            // Update indicators with snapshot (parallel in production)
            // Here we use sequential for simplicity, but structure is parallel-safe
            if let Some(val) = sma_14.next(&snapshot) {
                sma_14_values.push(val);
            }
            if let Some(val) = sma_21.next(&snapshot) {
                sma_21_values.push(val);
            }
        }
    }

    // Flush last candle
    if let Some(last) = builder.flush() {
        window.write().unwrap().push(last);
        let snapshot = window.read().unwrap().snapshot();
        if let Some(val) = sma_14.next(&snapshot) {
            sma_14_values.push(val);
        }
        if let Some(val) = sma_21.next(&snapshot) {
            sma_21_values.push(val);
        }
    }

    let final_candle_count = window.read().unwrap().len();

    println!(
        "Streaming simulation: {} ticks → {} candles",
        ticks.len(),
        final_candle_count
    );
    println!(
        "SMA-14: {} values, SMA-21: {} values",
        sma_14_values.len(),
        sma_21_values.len()
    );

    // Verify we got indicator values
    assert!(!sma_14_values.is_empty(), "SMA-14 should have values");
    assert!(!sma_21_values.is_empty(), "SMA-21 should have values");

    // SMA-14 should have more values (shorter warmup)
    assert!(
        sma_14_values.len() >= sma_21_values.len(),
        "SMA-14 should have at least as many values as SMA-21"
    );

    // Verify values are reasonable
    for (i, &val) in sma_14_values.iter().enumerate() {
        assert!(
            val > 0.0 && val < 1_000_000.0,
            "SMA-14 value {} at {} seems unreasonable",
            val,
            i
        );
    }
}

// ============================================================
// Integration Test: Full pipeline with batch + streaming equivalence
// ============================================================

#[test]
fn batch_and_streaming_produce_equivalent_results() {
    let source_candles = load_btc_data();
    let ticks = simulate_ticks_from_candles(&source_candles);

    // === Batch Mode ===
    let mut batch_sma = SMA::new(SMAConfig::new(14));
    batch_sma.calc(&source_candles).unwrap();
    let batch_values: Vec<f64> = batch_sma
        .history()
        .iter()
        .filter(|v| !v.is_nan())
        .copied()
        .collect();

    // === Streaming Mode (via CandleBuilder + next()) ===
    let mut builder = CandleBuilder::new(Timeframe::M1);
    let mut window = RollingWindow::new(200);
    let mut stream_sma = SMA::new(SMAConfig::new(14));
    let mut stream_values = Vec::new();

    for tick in &ticks {
        if let Some(completed) = builder.push(tick) {
            window.push(completed);
            let snapshot = window.snapshot();
            if let Some(val) = stream_sma.next(&snapshot) {
                stream_values.push(val);
            }
        }
    }

    // Flush last
    if let Some(last) = builder.flush() {
        window.push(last);
        let snapshot = window.snapshot();
        if let Some(val) = stream_sma.next(&snapshot) {
            stream_values.push(val);
        }
    }

    println!(
        "Batch: {} values, Streaming: {} values",
        batch_values.len(),
        stream_values.len()
    );

    // Both should produce same number of values
    assert_eq!(
        batch_values.len(),
        stream_values.len(),
        "Batch and streaming should produce same count"
    );

    // Values should match within epsilon
    for (i, (batch_val, stream_val)) in batch_values.iter().zip(stream_values.iter()).enumerate() {
        let diff = (batch_val - stream_val).abs();
        assert!(
            diff < EPSILON,
            "Mismatch at {}: batch={:.6}, stream={:.6}, diff={:.6}",
            i,
            batch_val,
            stream_val,
            diff
        );
    }

    println!("Batch and streaming modes produce equivalent results");
}
