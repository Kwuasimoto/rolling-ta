//! Volatility indicators.
//!
//! Indicators for measuring market volatility and price ranges.

mod tr;
mod atr;
mod bb;

pub use tr::TR;
pub use atr::ATR;
pub use bb::{BB, BBOutput};

// Re-export configs for convenience
pub use crate::ta::config::{TRConfig, ATRConfig, BBConfig};
