//! MFI reference tests.
//!
//! Verifies MFI (Money Flow Index) implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_hlcv, compare_values_with_warmup, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::volume::{MFI, MFIConfig};

#[test]
fn mfi_batch_vs_reference() {
    // btc-mfi.xlsx: timestamp(0), high(1), low(2), close(3), typical(4), volume(5),
    //               rmf(6), pmf(7), nmf(8), pmf_sum_14(9), nmf_sum_14(10), mfi(11)
    let cols = read_xlsx_by_position("resources/data/btc-mfi.xlsx", &[1, 2, 3, 5, 11]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let volumes = &cols[3];
    let expected_mfi = &cols[4];
    let candles = build_candles_hlcv(highs, lows, closes, volumes);

    // Batch calculation (validates against Python reference)
    // Python outputs zeros for warmup period; Rust only outputs computed values
    let period = 14;
    let mut mfi = MFI::new(MFIConfig::new(period));
    mfi.calc(&candles).unwrap();

    // Use warmup-aware comparison: skip first `period` values in expected data
    let comparisons = compare_values_with_warmup(
        "MFI batch vs reference",
        mfi.history(),
        expected_mfi,
        period,
        EPSILON,
    );
    println!(
        "MFI: {} values compared against Python reference",
        comparisons
    );
    assert!(
        comparisons > 100,
        "Should have compared many values, got {}",
        comparisons
    );
}

#[test]
fn mfi_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-mfi.xlsx", &[1, 2, 3, 5]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let volumes = &cols[3];
    let candles = build_candles_hlcv(highs, lows, closes, volumes);

    let period = 14;

    // 1. Batch calculation
    let mut batch = MFI::new(MFIConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = MFI::new(MFIConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only
    let batch_computed: Vec<f64> = batch.history().iter().filter(|&v| !v.is_nan()).copied().collect();
    let stream_computed = stream.history();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_histories_equal("MFI batch vs stream", &batch_computed, stream_computed, epsilon);
}

// NOTE: MFI does not support fixed-window testing.
//
// MFI requires accumulated state (rolling PMF/NMF sums based on typical price
// direction changes) that cannot be correctly computed from a fixed-size sliding
// window. The algorithm needs the history of typical price comparisons from the
// start of the data, not just the current window.
//
// Use streaming mode (next with growing snapshots) instead of fixed windows.
