//! RSI reference tests.
//!
//! Verifies RSI implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_from_closes, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::momentum::{RSIConfig, RSI};
use rolling_ta::prelude::*;

#[test]
fn rsi_batch_vs_reference() {
    // btc-rsi.xlsx: timestamp(0), close(1), ..., rsi(6)
    let cols = read_xlsx_by_position("resources/data/btc-rsi.xlsx", &[1, 6]);
    let closes = &cols[0];
    let expected = &cols[1];
    let candles = build_candles_from_closes(closes);

    // Batch calculation (validates against Python reference)
    let mut rsi = RSI::new(RSIConfig::new(14));
    rsi.calc(&candles).unwrap();

    let comparisons = compare_values("RSI batch vs reference", rsi.history(), expected, EPSILON);
    println!(
        "RSI: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn rsi_streaming_next_vs_batch() {
    // btc-rsi.xlsx: timestamp(0), close(1), ..., rsi(6)
    let cols = read_xlsx_by_position("resources/data/btc-rsi.xlsx", &[1, 6]);
    let closes = &cols[0];
    let candles = build_candles_from_closes(closes);

    let period = 14;

    // 1. Batch calculation - history includes NaN for warmup period
    let mut batch = RSI::new(RSIConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    // Each call to next() receives a growing snapshot
    // Note: next() only pushes to history when returning Some (not during warmup)
    let mut stream = RSI::new(RSIConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // History formats differ:
    // - batch.history(): [NaN × (period), val1, val2, ...] (length = candles.len())
    // - stream.history(): [val1, val2, ...] (length = candles.len() - period)
    //
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
    assert_histories_equal("RSI batch vs stream", &batch_computed, &stream_computed, epsilon);
}
