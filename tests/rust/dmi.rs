//! DMI reference tests.
//!
//! Verifies DMI (Directional Movement Index) implementation.
//! Tests batch mode and streaming mode (via next()).
//! DMI has compound output: plus_di, minus_di.
//! Note: No separate xlsx file for DMI - uses ADX data for +DI/-DI validation.

use crate::common::{
    assert_dmi_histories_equal, build_candles_hlc, compare_values, read_xlsx_by_position,
    EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{DMI, DMIConfig};

#[test]
fn dmi_batch_vs_reference() {
    // btc-adx.xlsx contains DMI data: +dmi(12), -dmi(13)
    let cols = read_xlsx_by_position("resources/data/btc-adx.xlsx", &[1, 2, 3, 12, 13]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let expected_plus_di = &cols[3];
    let expected_minus_di = &cols[4];
    let candles = build_candles_hlc(highs, lows, closes);

    // Batch calculation (validates against Python reference)
    let mut dmi = DMI::new(DMIConfig::new(14));
    dmi.calc(&candles).unwrap();

    let history = dmi.history();
    let rust_plus_di: Vec<f64> = history.iter().map(|o| o.plus_di).collect();
    let rust_minus_di: Vec<f64> = history.iter().map(|o| o.minus_di).collect();

    let plus_di_cmp = compare_values("+DI batch vs reference", &rust_plus_di, expected_plus_di, EPSILON);
    let minus_di_cmp = compare_values("-DI batch vs reference", &rust_minus_di, expected_minus_di, EPSILON);

    println!(
        "DMI: +DI={}, -DI={} values compared against Python reference",
        plus_di_cmp, minus_di_cmp
    );
    assert!(plus_di_cmp > 100, "Should have compared many +DI values, got {}", plus_di_cmp);
    assert!(minus_di_cmp > 100, "Should have compared many -DI values, got {}", minus_di_cmp);
}

#[test]
fn dmi_streaming_next_vs_batch() {
    let cols = read_xlsx_by_position("resources/data/btc-adx.xlsx", &[1, 2, 3]);
    let highs = &cols[0];
    let lows = &cols[1];
    let closes = &cols[2];
    let candles = build_candles_hlc(highs, lows, closes);

    let period = 14;

    // 1. Batch calculation
    let mut batch = DMI::new(DMIConfig::new(period));
    batch.calc(&candles).unwrap();

    // 2. Streaming via next()
    let mut stream = DMI::new(DMIConfig::new(period));
    for i in 1..=candles.len() {
        let snapshot = &candles[..i];
        stream.next(snapshot);
    }

    // Compare computed values only (skip NaN warmup from both)
    let batch_computed: Vec<_> = batch.history().iter().filter(|o| !o.plus_di.is_nan()).cloned().collect();
    let stream_computed: Vec<_> = stream.history().iter().filter(|o| !o.plus_di.is_nan()).cloned().collect();

    assert_eq!(
        batch_computed.len(),
        stream_computed.len(),
        "Computed value count should match: batch={}, stream={}",
        batch_computed.len(),
        stream_computed.len()
    );

    let epsilon = 1e-6;
    assert_dmi_histories_equal("DMI batch vs stream", &batch_computed, &stream_computed, epsilon);
}
