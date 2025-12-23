//! BOP reference tests.
//!
//! Verifies BOP (Balance of Power) implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_ohlc, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::momentum::{BOP, BOPConfig};
use rolling_ta::prelude::*;

#[test]
fn bop_batch_vs_reference() {
    // btc-bop.xlsx: timestamp(0), open(1), high(2), low(3), close(4), bop(5), bop_14(6)
    let cols = read_xlsx_by_position("resources/data/btc-bop.xlsx", &[1, 2, 3, 4, 6]);
    let opens = &cols[0];
    let highs = &cols[1];
    let lows = &cols[2];
    let closes = &cols[3];
    let expected_bop = &cols[4]; // bop_14 (smoothed)
    let candles = build_candles_ohlc(opens, highs, lows, closes);

    // Batch calculation (validates against Python reference)
    let mut bop = BOP::new(BOPConfig::new(14));
    bop.calc(&candles).unwrap();

    let comparisons = compare_values("BOP batch vs reference", bop.history(), expected_bop, EPSILON);
    println!(
        "BOP: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn bop_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-bop.xlsx", &[1, 2, 3, 4]);
    let opens = &cols[0];
    let highs = &cols[1];
    let lows = &cols[2];
    let closes = &cols[3];
    let candles = build_candles_ohlc(opens, highs, lows, closes);

    let period = 14;

    // 1. Batch calculation
    let mut batch = BOP::new(BOPConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = BOP::new(BOPConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only (skip NaN warmup from both)
    let batch_computed: Vec<f64> = batch.history().iter().filter(|&v| !v.is_nan()).copied().collect();
    let stream_computed: Vec<f64> = stream.history().iter().filter(|&v| !v.is_nan()).copied().collect();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_histories_equal("BOP batch vs stream", &batch_computed, &stream_computed, epsilon);
}

// Note: BOP uses internal SMA smoothing that accumulates state across calls,
// so fixed_window tests aren't applicable (similar to RSI, ATR with internal EMA).
