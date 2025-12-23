failures:

---- atr::atr_next_with_fixed_window stdout ----

thread 'atr::atr_next_with_fixed_window' (101628) panicked at tests\rust\atr.rs:109:9:
Should have result at index 15
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- bb::bb_streaming_next_vs_batch stdout ----

thread 'bb::bb_streaming_next_vs_batch' (60188) panicked at tests\rust\bb.rs:71:5:
assertion `left == right` failed: Computed value count should match: batch=181, stream=200
  left: 181
 right: 200

---- atr::atr_streaming_next_vs_batch stdout ----

thread 'atr::atr_streaming_next_vs_batch' (43636) panicked at tests\rust\atr.rs:77:5:
assertion `left == right` failed: Computed value count should match: batch=187, stream=200
  left: 187
 right: 200

---- adx::adx_next_with_fixed_window stdout ----

thread 'adx::adx_next_with_fixed_window' (28356) panicked at tests\rust\adx.rs:135:13:
ADX mismatch at index 42: +DI=(52.7905 vs 52.9799), -DI=(26.6647 vs 26.5707), DX=(32.8811 vs 33.1980), ADX=(27.6676 vs 27.6524)

---- adx::adx_streaming_next_vs_batch stdout ----

thread 'adx::adx_streaming_next_vs_batch' (106764) panicked at tests\rust\adx.rs:82:5:
assertion `left == right` failed: Computed value count should match: batch=187, stream=200
  left: 187
 right: 200

---- dmi::dmi_next_with_fixed_window stdout ----

thread 'dmi::dmi_next_with_fixed_window' (71956) panicked at tests\rust\dmi.rs:112:13:
DMI mismatch at index 28: +DI=(28.1374 vs 27.9826), -DI=(44.9399 vs 45.1756)

---- lr::linear_regression_forecast_streaming_next_vs_batch stdout ----

thread 'lr::linear_regression_forecast_streaming_next_vs_batch' (103276) panicked at tests\rust\lr.rs:199:5:
assertion `left == right` failed: Computed value count should match: batch=187, stream=200
  left: 187
 right: 200

---- obv::obv_next_with_fixed_window stdout ----

thread 'obv::obv_next_with_fixed_window' (119292) panicked at tests\rust\obv.rs:93:13:
OBV mismatch at index 4: Rust=19.093669, Expected=18.047314, rel_diff=0.057978

---- lr::linear_regression_streaming_next_vs_batch stdout ----

thread 'lr::linear_regression_streaming_next_vs_batch' (6120) panicked at tests\rust\lr.rs:77:5:
assertion `left == right` failed: Computed value count should match: batch=187, stream=200
  left: 187
 right: 200

---- lr::linear_regression_r2_streaming_next_vs_batch stdout ----

thread 'lr::linear_regression_r2_streaming_next_vs_batch' (58444) panicked at tests\rust\lr.rs:138:5:
assertion `left == right` failed: Computed value count should match: batch=187, stream=200
  left: 187
 right: 200

---- rsi::rsi_next_with_fixed_window stdout ----

thread 'rsi::rsi_next_with_fixed_window' (66056) panicked at tests\rust\rsi.rs:100:9:
Should have result at index 15

---- rsi::rsi_streaming_next_vs_batch stdout ----

thread 'rsi::rsi_streaming_next_vs_batch' (45136) panicked at tests\rust\rsi.rs:67:5:
assertion `left == right` failed: Computed value count should match: batch=186, stream=200
  left: 186
 right: 200

---- vwap::vwap_next_with_fixed_window stdout ----

thread 'vwap::vwap_next_with_fixed_window' (53360) panicked at tests\rust\vwap.rs:104:13:
VWAP mismatch at index 10: Rust=13301.410836, Expected=13307.660512, diff=6.249677


failures:
    adx::adx_next_with_fixed_window
    adx::adx_streaming_next_vs_batch
    atr::atr_next_with_fixed_window
    atr::atr_streaming_next_vs_batch
    bb::bb_streaming_next_vs_batch
    dmi::dmi_next_with_fixed_window
    lr::linear_regression_forecast_streaming_next_vs_batch
    lr::linear_regression_r2_streaming_next_vs_batch
    lr::linear_regression_streaming_next_vs_batch
    obv::obv_next_with_fixed_window
    rsi::rsi_next_with_fixed_window
    rsi::rsi_streaming_next_vs_batch
    vwap::vwap_next_with_fixed_window

test result: FAILED. 36 passed; 13 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.61s






# FIX

Please make it so indicators front fill their vectors with None where values cannot be calculated so these errors do not appear. It is okay if we use None placeholders or 0.0 even if a number cannot be calculated because the supplied slice's period is less than the indicators period.