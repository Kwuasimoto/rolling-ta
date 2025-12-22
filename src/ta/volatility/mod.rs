//! Volatility indicators.
//!
//! Indicators for measuring market volatility and price ranges.

mod atr;
mod bb;
mod donchian;
mod tr;

pub use atr::ATR;
pub use bb::{BB, BBOutput};
pub use donchian::{Donchian, DonchianOutput};
pub use tr::TR;

// Re-export configs for convenience
pub use crate::ta::config::{ATRConfig, BBConfig, DonchianConfig, TRConfig};