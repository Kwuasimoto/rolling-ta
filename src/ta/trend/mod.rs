//! Trend indicators.
//!
//! Indicators for identifying and following market trends.

mod sma;
// TODO: Migrate to new Indicator trait (calc(&[Ohlcv]), next(), no update())
// mod ema;
// mod wma;
// mod macd;
// mod hma;
// mod dmi;
// mod adx;
// mod lr;

pub use sma::SMA;
// TODO: Uncomment after migration
// pub use ema::EMA;
// pub use wma::WMA;
// pub use hma::HMA;
// pub use macd::{MACD, MACDOutput};
// pub use dmi::{DMI, DMIOutput};
// pub use adx::{ADX, ADXOutput};
// pub use lr::{LinearRegression, LinearRegressionR2, LinearRegressionForecast};

// Re-export configs for convenience
pub use crate::ta::config::SMAConfig;
// TODO: Uncomment after migration
// pub use crate::ta::config::{EMAConfig, WMAConfig, MACDConfig, HMAConfig, DMIConfig, ADXConfig, LinearRegressionConfig};
