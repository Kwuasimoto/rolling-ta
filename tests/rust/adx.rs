//! ADX reference tests.
//!
//! Verifies ADX implementation against Python rolling-ta output.
//! Tests batch mode and streaming mode (via next()).
//! ADX has compound output: plus_di, minus_di, dx, adx.

use crate::common::{
    assert_adx_histories_equal, build_candles_hlc, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{ADX, ADXConfig};

#[test]
fn adx_batch_vs_reference() {
    // btc-adx.xlsx: timestamp(0), high(1), low(2), close(3), ..., +dmi(12), -dmi(13), dx(14), adx(15)
    let cols = read_xlsx_by_position(
        "resources/data/btc-adx.xlsx",
        &[1, 2, 3, 12, 13, 14, 15],
    );
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_plus_di = &cols[3];
    let expected_minus_di = &cols[4];
    let expected_dx = &cols[5];
    let expected_adx = &cols[6];
    let candles = build_candles_hlc(highs, lows, closes);

    // Batch calculation (validates against Python reference)
    let mut adx = ADX::new(ADXConfig::new(14, 14));
    adx.calc(&candles).unwrap();

    let history = adx.history();
    let rust_plus_di: Vec<f64> = history.iter().map(|o| o.plus_di).collect();
    let rust_minus_di: Vec<f64> = history.iter().map(|o| o.minus_di).collect();
    let rust_dx: Vec<f64> = history.iter().map(|o| o.dx).collect();
    let rust_adx: Vec<f64> = history.iter().map(|o| o.adx).collect();

    let plus_di_cmp = compare_values("+DI batch vs reference", &rust_plus_di, expected_plus_di, EPSILON);
    let minus_di_cmp = compare_values("-DI batch vs reference", &rust_minus_di, expected_minus_di, EPSILON);
    let dx_cmp = compare_values("DX batch vs reference", &rust_dx, expected_dx, EPSILON);
    let adx_cmp = compare_values("ADX batch vs reference", &rust_adx, expected_adx, EPSILON);

    println!(
        "ADX: +DI={}, -DI={}, DX={}, ADX={} values compared against Python reference",
        plus_di_cmp, minus_di_cmp, dx_cmp, adx_cmp
    );
    assert!(plus_di_cmp > 100, "Should have compared many +DI values, got {}", plus_di_cmp);
    assert!(minus_di_cmp > 100, "Should have compared many -DI values, got {}", minus_di_cmp);
    assert!(dx_cmp > 100, "Should have compared many DX values, got {}", dx_cmp);
    assert!(adx_cmp > 100, "Should have compared many ADX values, got {}", adx_cmp);
}

#[test]
fn adx_streaming_next_vs_batch() {
    // btc-adx.xlsx: timestamp(0), high(1), low(2), close(3), ...
    let cols = read_xlsx_by_position("resources/data/btc-adx.xlsx", &[1, 2, 3]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let candles = build_candles_hlc(highs, lows, closes);

    let dmi_period = 14;
    let adx_period = 14;

    // 1. Batch calculation
    let mut batch = ADX::new(ADXConfig::new(dmi_period, adx_period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next() - simulates SharedWindow snapshot pattern
    let mut stream = ADX::new(ADXConfig::new(dmi_period, adx_period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only (skip NaN warmup from both)
    let batch_computed: Vec<_> = batch.history().iter().filter(|o| !o.adx.is_nan()).cloned().collect();
    let stream_computed: Vec<_> = stream.history().iter().filter(|o| !o.adx.is_nan()).cloned().collect();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_adx_histories_equal("ADX batch vs stream", &batch_computed, &stream_computed, epsilon);
}
