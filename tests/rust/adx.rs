//! ADX reference tests.
//!
//! Verifies ADX implementation against Python rolling-ta output.
//! Tests batch, streaming, and hybrid modes in a single consolidated test.
//! ADX has compound output: plus_di, minus_di, dx, adx.

use crate::common::{
    assert_adx_histories_equal, build_ohlcv_hlc, compare_values, read_xlsx_by_position,
    slice_ohlcv, tick_at, EPSILON,
};
use rolling_ta::prelude::*;
use rolling_ta::trend::{ADX, ADXConfig};

#[test]
fn adx_modes_equivalent() {
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
    let data = build_ohlcv_hlc(highs, lows, closes);

    // 1. Batch calculation - validate all components against Python reference
    let mut batch = ADX::new(ADXConfig::new(14, 14));
    batch.calc(&data).unwrap();

    let history = batch.history();
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

    // 2. Full streaming (tests update() from scratch, including warmup)
    let mut stream = ADX::new(ADXConfig::new(14, 14));
    for i in 0..data.len() {
        stream.update(&tick_at(&data, i)).unwrap();
    }

    // 3. Hybrid: 50% batch + 50% streaming (tests calc-to-update transition)
    let half = data.len() / 2;
    let mut hybrid = ADX::new(ADXConfig::new(14, 14));
    hybrid.calc(&slice_ohlcv(&data, 0, half)).unwrap();
    for i in half..data.len() {
        hybrid.update(&tick_at(&data, i)).unwrap();
    }

    // All three modes must produce identical results
    let epsilon = 1e-6;
    assert_adx_histories_equal("ADX batch vs stream", batch.history(), stream.history(), epsilon);
    assert_adx_histories_equal("ADX batch vs hybrid", batch.history(), hybrid.history(), epsilon);
}
