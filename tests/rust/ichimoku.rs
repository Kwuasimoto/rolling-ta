//! Ichimoku Cloud reference tests.
//!
//! Verifies Ichimoku implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! Ichimoku has compound output: tenkan, kijun, senkou_a, senkou_b.

use crate::common::{
    assert_ichimoku_histories_equal, build_candles_hlc, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{Ichimoku, IchimokuConfig};

#[test]
fn ichimoku_batch_vs_reference() {
    // btc-ichimoku_cloud.xlsx: timestamp(0), high(1), low(2), high_max_9(3), low_min_9(4),
    //                          high_max_26(5), low_max_26(6), high_max_52(7), low_max_52(8),
    //                          tenkan(9), kijun(10), senkou_a(11), senkou_b(12)
    let cols = read_xlsx_by_position(
        "resources/data/btc-ichimoku_cloud.xlsx",
        &[1, 2, 9, 10, 11, 12],
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let expected_tenkan = &cols[2];
    let expected_kijun = &cols[3];
    let expected_senkou_a = &cols[4];
    let expected_senkou_b = &cols[5];
    let candles = build_candles_hlc(highs, lows, highs); // use high as close (not used)

    // Batch calculation (validates against Python reference)
    let config = IchimokuConfig {
        tenkan_period: 9,
        kijun_period: 26,
        senkou_b_period: 52,
        displacement: 26,
    };
    let mut ichimoku = Ichimoku::new(config);
    ichimoku.calc(&candles).unwrap();

    let history = ichimoku.history();
    let rust_tenkan: Vec<f64> = history.iter().map(|o| o.tenkan).collect();
    let rust_kijun: Vec<f64> = history.iter().map(|o| o.kijun).collect();
    let rust_senkou_a: Vec<f64> = history.iter().map(|o| o.senkou_a).collect();
    let rust_senkou_b: Vec<f64> = history.iter().map(|o| o.senkou_b).collect();

    let tenkan_cmp = compare_values("Ichimoku tenkan vs reference", &rust_tenkan, expected_tenkan, EPSILON);
    let kijun_cmp = compare_values("Ichimoku kijun vs reference", &rust_kijun, expected_kijun, EPSILON);
    let senkou_a_cmp = compare_values("Ichimoku senkou_a vs reference", &rust_senkou_a, expected_senkou_a, EPSILON);
    let senkou_b_cmp = compare_values("Ichimoku senkou_b vs reference", &rust_senkou_b, expected_senkou_b, EPSILON);

    println!(
        "Ichimoku: tenkan={}, kijun={}, senkou_a={}, senkou_b={} values compared against Python reference",
        tenkan_cmp, kijun_cmp, senkou_a_cmp, senkou_b_cmp
    );
    assert!(tenkan_cmp > 100, "Should have compared many tenkan values, got {}", tenkan_cmp);
    assert!(kijun_cmp > 100, "Should have compared many kijun values, got {}", kijun_cmp);
    assert!(senkou_a_cmp > 100, "Should have compared many senkou_a values, got {}", senkou_a_cmp);
    assert!(senkou_b_cmp > 100, "Should have compared many senkou_b values, got {}", senkou_b_cmp);
}

#[test]
fn ichimoku_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-ichimoku_cloud.xlsx", &[1, 2]);
    let highs = &cols[0];
    let lows = &cols[1];
    let candles = build_candles_hlc(highs, lows, highs);

    // 1. Batch calculation
    let config = IchimokuConfig {
        tenkan_period: 9,
        kijun_period: 26,
        senkou_b_period: 52,
        displacement: 26,
    };
    let mut batch = Ichimoku::new(config.clone());
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = Ichimoku::new(config);
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only
    let batch_computed: Vec<_> = batch.history().iter().filter(|o| !o.senkou_b.is_nan()).cloned().collect();
    let stream_computed = stream.history();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_ichimoku_histories_equal("Ichimoku batch vs stream", &batch_computed, stream_computed, epsilon);
}

#[test]
fn ichimoku_next_with_fixed_window() {
    let cols = read_xlsx_by_position(
        "resources/data/btc-ichimoku_cloud.xlsx",
        &[1, 2, 9, 10, 11, 12],
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let expected_tenkan = &cols[2];
    let expected_kijun = &cols[3];
    let expected_senkou_a = &cols[4];
    let expected_senkou_b = &cols[5];
    let candles = build_candles_hlc(highs, lows, highs);

    // Ichimoku needs senkou_b_period (52) candles minimum
    let window_size = 52;
    let config = IchimokuConfig {
        tenkan_period: 9,
        kijun_period: 26,
        senkou_b_period: 52,
        displacement: 26,
    };
    let mut ichimoku = Ichimoku::new(config);

    // Feed snapshots of fixed window size
    for i in window_size..candles.len() {
        let snapshot = &candles[i - window_size + 1..=i];
        let result = ichimoku.next(snapshot);

        assert!(result.is_some(), "Should have result at index {}", i);

        let output = result.unwrap();
        let true_tenkan = expected_tenkan[i];
        let true_kijun = expected_kijun[i];
        let true_senkou_a = expected_senkou_a[i];
        let true_senkou_b = expected_senkou_b[i];

        if !output.senkou_b.is_nan() && !true_senkou_b.is_nan() {
            let diff_tenkan = (output.tenkan - true_tenkan).abs();
            let diff_kijun = (output.kijun - true_kijun).abs();
            let diff_senkou_a = (output.senkou_a - true_senkou_a).abs();
            let diff_senkou_b = (output.senkou_b - true_senkou_b).abs();

            assert!(
                diff_tenkan < EPSILON && diff_kijun < EPSILON && diff_senkou_a < EPSILON && diff_senkou_b < EPSILON,
                "Ichimoku mismatch at index {}: tenkan=({:.4} vs {:.4}), kijun=({:.4} vs {:.4}), senkou_a=({:.4} vs {:.4}), senkou_b=({:.4} vs {:.4})",
                i, output.tenkan, true_tenkan, output.kijun, true_kijun,
                output.senkou_a, true_senkou_a, output.senkou_b, true_senkou_b
            );
        }
    }
}
