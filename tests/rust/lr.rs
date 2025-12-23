//! Linear Regression reference tests.
//!
//! Verifies LinearRegression, LinearRegressionR2, and LinearRegressionForecast
//! implementations against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).

use crate::common::{
    assert_histories_equal, build_candles_hlc, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{
    LinearRegression, LinearRegressionConfig, LinearRegressionForecast, LinearRegressionR2,
};

#[test]
fn linear_regression_batch_vs_reference() {
    // btc-linear_regression.xlsx columns:
    // timestamp(0), high(1), low(2), close(3), typical(4), row(5), intercept(6), slope(7), lr2(8), forecast(9)
    let cols = read_xlsx_by_position(
        "resources/data/btc-linear_regression.xlsx",
        &[1, 2, 3, 6, 7], // high, low, close, intercept, slope
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_intercept = &cols[3];
    let expected_slope = &cols[4];
    let candles = build_candles_hlc(highs, lows, closes);

    // LinearRegression outputs: slope * (period - 1) + intercept
    let period = 14;
    let mut expected_lr: Vec<f64> = vec![f64::NAN; candles.len()];
    for i in (period - 1)..candles.len() {
        if !expected_slope[i].is_nan() && !expected_intercept[i].is_nan() {
            expected_lr[i] = expected_slope[i] * (period - 1) as f64 + expected_intercept[i];
        }
    }

    // Batch calculation (validates against Python reference)
    let mut lr = LinearRegression::new(LinearRegressionConfig::new(14));
    lr.calc(&candles).unwrap();

    let comparisons = compare_values("LR batch vs reference", lr.history(), &expected_lr, EPSILON);
    println!("LR: {} values compared against Python reference", comparisons);
    assert!(comparisons > 100, "Should have compared many LR values, got {}", comparisons);
}

#[test]
fn linear_regression_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position(
        "resources/data/btc-linear_regression.xlsx",
        &[1, 2, 3], // high, low, close
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let candles = build_candles_hlc(highs, lows, closes);

    let period = 14;

    // 1. Batch calculation
    let mut batch = LinearRegression::new(LinearRegressionConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = LinearRegression::new(LinearRegressionConfig::new(period));
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
    assert_histories_equal("LR batch vs stream", &batch_computed, &stream_computed, epsilon);
}

#[test]
fn linear_regression_r2_batch_vs_reference() {
    let cols = read_xlsx_by_position(
        "resources/data/btc-linear_regression.xlsx",
        &[1, 2, 3, 8], // high, low, close, lr2
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_lr2 = &cols[3];
    let candles = build_candles_hlc(highs, lows, closes);

    // Batch calculation (validates against Python reference)
    let mut lr2 = LinearRegressionR2::new(LinearRegressionConfig::new(14));
    lr2.calc(&candles).unwrap();

    let comparisons = compare_values("LR2 batch vs reference", lr2.history(), expected_lr2, EPSILON);
    println!("LR2: {} values compared against Python reference", comparisons);
    assert!(comparisons > 100, "Should have compared many LR2 values, got {}", comparisons);
}

#[test]
fn linear_regression_r2_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position(
        "resources/data/btc-linear_regression.xlsx",
        &[1, 2, 3], // high, low, close
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let candles = build_candles_hlc(highs, lows, closes);

    let period = 14;

    // 1. Batch calculation
    let mut batch = LinearRegressionR2::new(LinearRegressionConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = LinearRegressionR2::new(LinearRegressionConfig::new(period));
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
    assert_histories_equal("LR2 batch vs stream", &batch_computed, &stream_computed, epsilon);
}

#[test]
fn linear_regression_forecast_batch_vs_reference() {
    let cols = read_xlsx_by_position(
        "resources/data/btc-linear_regression.xlsx",
        &[1, 2, 3, 9], // high, low, close, forecast
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_lrf = &cols[3];
    let candles = build_candles_hlc(highs, lows, closes);

    // Batch calculation (validates against Python reference)
    let mut lrf = LinearRegressionForecast::new(LinearRegressionConfig::new(14));
    lrf.calc(&candles).unwrap();

    let comparisons = compare_values("LRF batch vs reference", lrf.history(), expected_lrf, EPSILON);
    println!("LRF: {} values compared against Python reference", comparisons);
    assert!(comparisons > 100, "Should have compared many LRF values, got {}", comparisons);
}

#[test]
fn linear_regression_forecast_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position(
        "resources/data/btc-linear_regression.xlsx",
        &[1, 2, 3], // high, low, close
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let candles = build_candles_hlc(highs, lows, closes);

    let period = 14;

    // 1. Batch calculation
    let mut batch = LinearRegressionForecast::new(LinearRegressionConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = LinearRegressionForecast::new(LinearRegressionConfig::new(period));
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
    assert_histories_equal("LRF batch vs stream", &batch_computed, &stream_computed, epsilon);
}

#[test]
fn linear_regression_optimized_path() {
    // Test the calc_from_models() optimization path
    let cols = read_xlsx_by_position(
        "resources/data/btc-linear_regression.xlsx",
        &[1, 2, 3, 8, 9], // high, low, close, lr2, forecast
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_lr2 = &cols[3];
    let expected_lrf = &cols[4];
    let candles = build_candles_hlc(highs, lows, closes);

    // Calculate LR once
    let mut lr = LinearRegression::new(LinearRegressionConfig::new(14));
    lr.calc(&candles).unwrap();

    // Reuse models for LR2
    let mut lr2_optimized = LinearRegressionR2::new(LinearRegressionConfig::new(14));
    lr2_optimized.calc_from_models(&candles, lr.models()).unwrap();

    // Compare against independent calculation
    let mut lr2_independent = LinearRegressionR2::new(LinearRegressionConfig::new(14));
    lr2_independent.calc(&candles).unwrap();

    let epsilon = 1e-10;
    assert_histories_equal("LR2 optimized vs independent", lr2_optimized.history(), lr2_independent.history(), epsilon);

    // Verify against reference
    let lr2_cmp = compare_values("LR2 (optimized) vs reference", lr2_optimized.history(), expected_lr2, EPSILON);
    assert!(lr2_cmp > 100, "Should have compared many LR2 values, got {}", lr2_cmp);

    // Reuse models for LRF
    let mut lrf_optimized = LinearRegressionForecast::new(LinearRegressionConfig::new(14));
    lrf_optimized.calc_from_models(lr.models()).unwrap();

    // Compare against independent calculation
    let mut lrf_independent = LinearRegressionForecast::new(LinearRegressionConfig::new(14));
    lrf_independent.calc(&candles).unwrap();

    assert_histories_equal("LRF optimized vs independent", lrf_optimized.history(), lrf_independent.history(), epsilon);

    // Verify against reference
    let lrf_cmp = compare_values("LRF (optimized) vs reference", lrf_optimized.history(), expected_lrf, EPSILON);
    assert!(lrf_cmp > 100, "Should have compared many LRF values, got {}", lrf_cmp);
}
