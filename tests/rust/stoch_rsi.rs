//! StochRSI reference tests.
//!
//! Verifies StochRSI implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! StochRSI has compound output: k (stochastic), d (signal).

use crate::common::{
    assert_stochrsi_histories_equal, build_candles_from_closes, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::momentum::{StochRSI, StochRSIConfig};
use rolling_ta::prelude::*;

#[test]
fn stochrsi_batch_vs_reference() {
    // btc-rsi.xlsx contains StochRSI data:
    // timestamp(0), close(1), gain(2), loss(3), gain_14(4), loss_14(5),
    // rsi(6), rsi_min_14(7), rsi_max_14(8), stoch_rsi(9), stoch_k(10), stoch_d(11)
    let cols = read_xlsx_by_position("resources/data/btc-rsi.xlsx", &[1, 10, 11]);
    let closes = &cols[0];
    let expected_k = &cols[1];
    let expected_d = &cols[2];
    let candles = build_candles_from_closes(closes);

    // Batch calculation (validates against Python reference)
    // Python defaults: rsi=14, stoch_rsi=10, stoch_k=3, stoch_d=3
    let rsi_period = 14;
    let stoch_period = 10;
    let k_smoothing = 3;
    let d_smoothing = 3;
    let mut stochrsi = StochRSI::new(StochRSIConfig::new(rsi_period, stoch_period, k_smoothing, d_smoothing));
    stochrsi.calc(&candles).unwrap();

    // History is same length as input data (includes zeros for warmup)
    let history = stochrsi.history();
    assert_eq!(history.len(), candles.len(), "History should match input length");

    let rust_k: Vec<f64> = history.iter().map(|o| o.k).collect();
    let rust_d: Vec<f64> = history.iter().map(|o| o.d).collect();

    // Compare directly - both should have zeros for warmup, then valid values
    let mut k_comparisons = 0;
    let mut d_comparisons = 0;

    for i in 0..rust_k.len().min(expected_k.len()) {
        let rk = rust_k[i];
        let ek = expected_k[i];
        let rd = rust_d[i];
        let ed = expected_d[i];

        // Skip comparison if both are zero (warmup) or NaN
        if (rk == 0.0 && ek == 0.0) || rk.is_nan() || ek.is_nan() {
            continue;
        }

        let diff_k = (rk - ek).abs();
        assert!(
            diff_k < EPSILON,
            "StochRSI K mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}",
            i, rk, ek, diff_k
        );
        k_comparisons += 1;

        if (rd == 0.0 && ed == 0.0) || rd.is_nan() || ed.is_nan() {
            continue;
        }

        let diff_d = (rd - ed).abs();
        assert!(
            diff_d < EPSILON,
            "StochRSI D mismatch at index {}: Rust={:.6}, Expected={:.6}, diff={:.6}",
            i, rd, ed, diff_d
        );
        d_comparisons += 1;
    }

    println!(
        "StochRSI: K={}, D={} values compared against Python reference",
        k_comparisons, d_comparisons
    );
    assert!(k_comparisons > 100, "Should have compared many K values, got {}", k_comparisons);
    assert!(d_comparisons > 100, "Should have compared many D values, got {}", d_comparisons);
}

#[test]
fn stochrsi_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-rsi.xlsx", &[1]);
    let closes = &cols[0];
    let candles = build_candles_from_closes(closes);

    // Use Python default parameters
    let rsi_period = 14;
    let stoch_period = 10;
    let k_smoothing = 3;
    let d_smoothing = 3;

    // 1. Batch calculation
    let mut batch = StochRSI::new(StochRSIConfig::new(rsi_period, stoch_period, k_smoothing, d_smoothing));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - with all candles at once (like joining mid-stream)
    let mut stream = StochRSI::new(StochRSIConfig::new(rsi_period, stoch_period, k_smoothing, d_smoothing));
    stream.next(&candles);

    // Compare histories directly
    assert_eq!(
        batch.len(),
        stream.len(),
        "History lengths should match: batch={}, stream={}",
        batch.len(),
        stream.len()
    );

    let epsilon = 1e-6;
    assert_stochrsi_histories_equal("StochRSI batch vs stream", batch.history(), stream.history(), epsilon);
}

// NOTE: StochRSI does not support fixed-window testing.
//
// StochRSI uses backward smoothing which requires all values to be present.
// This is fundamentally incompatible with fixed-size sliding windows.
// Use streaming mode (next with growing snapshots) instead.
