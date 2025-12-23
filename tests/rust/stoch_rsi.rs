//! StochRSI reference tests.
//!
//! Verifies StochRSI implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! StochRSI has compound output: k (stochastic), d (signal).

use crate::common::{
    assert_stochrsi_histories_equal, build_candles_from_closes, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::momentum::{StochRSI, StochRSIConfig};
use rolling_ta::prelude::*;

#[test]
fn stochrsi_batch_vs_reference() {
    // btc-rsi.xlsx contains StochRSI data:
    // timestamp(0), close(1), ..., stoch_rsi(9), stoch_k(10), stoch_d(11)
    let cols = read_xlsx_by_position("resources/data/btc-rsi.xlsx", &[1, 10, 11]);
    let closes = &cols[0];
    let expected_k = &cols[1];
    let expected_d = &cols[2];
    let candles = build_candles_from_closes(closes);

    // Batch calculation (validates against Python reference)
    // StochRSI uses: rsi_period=14, stoch_period=14, k_period=3, d_period=3
    let mut stochrsi = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));
    stochrsi.calc(&candles).unwrap();

    let history = stochrsi.history();
    let rust_k: Vec<f64> = history.iter().map(|o| o.k).collect();
    let rust_d: Vec<f64> = history.iter().map(|o| o.d).collect();

    let k_cmp = compare_values("StochRSI K vs reference", &rust_k, expected_k, EPSILON);
    let d_cmp = compare_values("StochRSI D vs reference", &rust_d, expected_d, EPSILON);

    println!(
        "StochRSI: K={}, D={} values compared against Python reference",
        k_cmp, d_cmp
    );
    assert!(k_cmp > 100, "Should have compared many K values, got {}", k_cmp);
    assert!(d_cmp > 100, "Should have compared many D values, got {}", d_cmp);
}

#[test]
fn stochrsi_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-rsi.xlsx", &[1]);
    let closes = &cols[0];
    let candles = build_candles_from_closes(closes);

    // 1. Batch calculation
    let mut batch = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = StochRSI::new(StochRSIConfig::new(14, 14, 3, 3));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only (skip NaN warmup from batch)
    let batch_computed: Vec<_> = batch.history().iter().filter(|o| !o.k.is_nan()).cloned().collect();
    let stream_computed = stream.history();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_stochrsi_histories_equal("StochRSI batch vs stream", &batch_computed, stream_computed, epsilon);
}

#[test]
fn stochrsi_next_with_fixed_window() {
    let cols = read_xlsx_by_position("resources/data/btc-rsi.xlsx", &[1, 10, 11]);
    let closes = &cols[0];
    let expected_k = &cols[1];
    let expected_d = &cols[2];
    let candles = build_candles_from_closes(closes);

    // StochRSI needs: rsi_period + stoch_period + k_period + d_period candles
    let rsi_period = 14;
    let stoch_period = 14;
    let k_period = 3;
    let d_period = 3;
    let window_size = rsi_period + stoch_period + k_period + d_period;
    let mut stochrsi = StochRSI::new(StochRSIConfig::new(rsi_period, stoch_period, k_period, d_period));

    // Feed snapshots of fixed window size
    for i in window_size..candles.len() {
        let snapshot = &candles[i - window_size + 1..=i];
        let result = stochrsi.next(snapshot);

        assert!(result.is_some(), "Should have result at index {}", i);

        let output = result.unwrap();
        let true_k = expected_k[i];
        let true_d = expected_d[i];

        if !output.k.is_nan() && !true_k.is_nan() {
            let diff_k = (output.k - true_k).abs();
            let diff_d = (output.d - true_d).abs();

            assert!(
                diff_k < EPSILON && diff_d < EPSILON,
                "StochRSI mismatch at index {}: K=({:.4} vs {:.4}), D=({:.4} vs {:.4})",
                i, output.k, true_k, output.d, true_d
            );
        }
    }
}
