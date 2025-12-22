//! Trend indicators.
//!
//! Indicators for identifying and following market trends.

mod adx;
mod dmi;
mod ema;
mod hma;
mod ichimoku;
mod lr;
mod macd;
mod sma;
mod wma;

pub use adx::{ADXOutput, ADX};
pub use dmi::{DMIOutput, DMI};
pub use ema::EMA;
pub use hma::HMA;
pub use ichimoku::{Ichimoku, IchimokuOutput};
pub use lr::{LinearRegression, LinearRegressionForecast, LinearRegressionR2};
pub use macd::{MACDOutput, MACD};
pub use sma::SMA;
pub use wma::WMA;

// Re-export configs for convenience
pub use crate::ta::config::{
    ADXConfig, DMIConfig, EMAConfig, HMAConfig, IchimokuConfig, LinearRegressionConfig, MACDConfig,
    SMAConfig, WMAConfig,
};
