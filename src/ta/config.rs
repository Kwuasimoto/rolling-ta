//! Configuration structs for all indicators.
//!
//! Each indicator has a dedicated config struct with sensible defaults.

// ============================================================
// Trend Indicators
// ============================================================

/// Simple Moving Average configuration.
///
/// SMA computes directly from `&[Ohlcv]` slices and does not own a window.
/// Temporal candle management is handled by `SharedWindow`, not the indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SMAConfig {
    pub period: usize,
}

impl Default for SMAConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl SMAConfig {
    /// Create a new SMA config with period.
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

/// Exponential Moving Average configuration.
///
/// EMA computes directly from `&[Ohlcv]` slices and does not own a window.
/// Temporal candle management is handled by `CandleBuilder`, not the indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EMAConfig {
    pub period: usize,
}

impl Default for EMAConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl EMAConfig {
    /// Create a new EMA config with period.
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// EMA multiplier: 2 / (period + 1)
    #[inline]
    pub fn multiplier(&self) -> f64 {
        2.0 / (self.period as f64 + 1.0)
    }
}

/// Weighted Moving Average configuration.
///
/// WMA computes directly from `&[Ohlcv]` slices and does not own a window.
/// Temporal candle management is handled by `CandleBuilder`, not the indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WMAConfig {
    pub period: usize,
}

impl Default for WMAConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl WMAConfig {
    /// Create a new WMA config with period.
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Sum of weights: period * (period + 1) / 2
    #[inline]
    pub fn weight_sum(&self) -> usize {
        self.period * (self.period + 1) / 2
    }
}

/// Hull Moving Average configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HMAConfig {
    pub period: usize,
}

impl Default for HMAConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl HMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Half period for HMA calculation.
    #[inline]
    pub fn half_period(&self) -> usize {
        self.period / 2
    }

    /// Square root period for final WMA.
    #[inline]
    pub fn sqrt_period(&self) -> usize {
        (self.period as f64).sqrt().floor() as usize
    }
}

/// MACD configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MACDConfig {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
}

impl Default for MACDConfig {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}

impl MACDConfig {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            signal_period,
        }
    }
}

// ============================================================
// Momentum Indicators
// ============================================================

/// Relative Strength Index configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RSIConfig {
    pub period: usize,
}

impl Default for RSIConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl RSIConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

/// Stochastic RSI configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StochRSIConfig {
    pub rsi_period: usize,
    pub stoch_period: usize,
    pub k_smoothing: usize,
    pub d_smoothing: usize,
}

impl StochRSIConfig {
    pub fn new(rsi_period: usize, stoch_period: usize, k_smoothing: usize, d_smoothing: usize) -> Self {
        Self {
            rsi_period,
            stoch_period,
            k_smoothing,
            d_smoothing,
        }
    }
}

impl Default for StochRSIConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            stoch_period: 14,
            k_smoothing: 3,
            d_smoothing: 3,
        }
    }
}

/// Balance of Power configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BOPConfig {
    /// Smoothing period (SMA). 0 or 1 means no smoothing.
    pub smoothing: usize,
}

impl BOPConfig {
    pub fn new(smoothing: usize) -> Self {
        Self { smoothing }
    }
}

impl Default for BOPConfig {
    fn default() -> Self {
        Self { smoothing: 14 }
    }
}

/// Rate of Change (ROC) configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ROCConfig {
    pub period: usize,
}

impl Default for ROCConfig {
    fn default() -> Self {
        Self { period: 12 }
    }
}

impl ROCConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

// ============================================================
// Volatility Indicators
// ============================================================

/// True Range configuration (no parameters).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TRConfig;

/// Average True Range configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ATRConfig {
    pub period: usize,
}

impl Default for ATRConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl ATRConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

/// Bollinger Bands configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBConfig {
    pub period: usize,
    pub std_dev_mult: f64,
}

impl Default for BBConfig {
    fn default() -> Self {
        Self {
            period: 20,
            std_dev_mult: 2.0,
        }
    }
}

impl BBConfig {
    pub fn new(period: usize, std_dev_mult: f64) -> Self {
        Self { period, std_dev_mult }
    }
}

/// Donchian Channels configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DonchianConfig {
    pub period: usize,
}

impl DonchianConfig {
    /// Create a new Donchian configuration.
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for DonchianConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

// ============================================================
// Volume Indicators
// ============================================================

/// On-Balance Volume configuration (no parameters).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OBVConfig;

/// Money Flow Index configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MFIConfig {
    pub period: usize,
}

impl MFIConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for MFIConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

/// Chaikin Money Flow configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CMFConfig {
    pub period: usize,
}

impl CMFConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for CMFConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// VWAP configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VWAPConfig {
    /// Reset interval in seconds (86400 = daily, 0 = no reset).
    pub reset_interval: i64,
}

impl Default for VWAPConfig {
    fn default() -> Self {
        Self {
            reset_interval: 86400,
        }
    }
}

impl VWAPConfig {
    /// Create a new VWAP configuration.
    ///
    /// # Arguments
    ///
    /// * `reset_interval` - Reset interval in seconds (86400 = daily, 0 = no reset)
    pub fn new(reset_interval: i64) -> Self {
        Self { reset_interval }
    }
}

// ============================================================
// Trend Strength Indicators
// ============================================================

/// DMI (Directional Movement Indicator) configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DMIConfig {
    pub period: usize,
}

impl Default for DMIConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl DMIConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

/// ADX (Average Directional Index) configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ADXConfig {
    pub dmi_period: usize,
    pub adx_period: usize,
}

impl Default for ADXConfig {
    fn default() -> Self {
        Self {
            dmi_period: 14,
            adx_period: 14,
        }
    }
}

impl ADXConfig {
    pub fn new(dmi_period: usize, adx_period: usize) -> Self {
        Self {
            dmi_period,
            adx_period,
        }
    }
}

// ============================================================
// Linear Regression
// ============================================================

/// Linear Regression configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearRegressionConfig {
    pub period: usize,
    pub forecast: usize,
}

impl Default for LinearRegressionConfig {
    fn default() -> Self {
        Self {
            period: 14,
            forecast: 0,
        }
    }
}

impl LinearRegressionConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            forecast: 0,
        }
    }
}

// ============================================================
// Ichimoku Cloud
// ============================================================

/// Ichimoku Cloud configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IchimokuConfig {
    pub tenkan_period: usize,
    pub kijun_period: usize,
    pub senkou_b_period: usize,
    pub displacement: usize,
}

impl Default for IchimokuConfig {
    fn default() -> Self {
        Self {
            tenkan_period: 9,
            kijun_period: 26,
            senkou_b_period: 52,
            displacement: 26,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_multiplier() {
        let config = EMAConfig::new(14);
        let mult = config.multiplier();
        // 2 / (14 + 1) = 2/15 ≈ 0.1333
        assert!((mult - 0.1333).abs() < 0.001);
    }

    #[test]
    fn wma_weight_sum() {
        let config = WMAConfig::new(5);
        // 5 * 6 / 2 = 15
        assert_eq!(config.weight_sum(), 15);
    }

    #[test]
    fn hma_periods() {
        let config = HMAConfig::new(16);
        assert_eq!(config.half_period(), 8);
        assert_eq!(config.sqrt_period(), 4);
    }
}
