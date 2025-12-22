//! Indicator Manager for orchestrating multiple indicators with a shared window.
//!
//! The `IndicatorManager` provides:
//! - Centralized lifecycle management for indicators
//! - Parallel execution via Rayon
//! - Atomic replacement for config changes (immutable replace pattern)
//! - Type-safe indicator access via handles
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    IndicatorManager                              │
//! │                                                                  │
//! │  ┌─────────────┐    ┌──────────────────────────────────────┐    │
//! │  │ SharedWindow │    │ HashMap<IndicatorId, Box<DynIndicator>>│  │
//! │  │ Arc<RwLock<>>│    │   id_1 → SMA                         │    │
//! │  └──────┬──────┘    │   id_2 → EMA                         │    │
//! │         │           └──────────────────────────────────────┘    │
//! │         │ snapshot()              │ par_iter_mut()              │
//! │         ▼                         ▼                              │
//! │    Vec<Ohlcv> ──────────────► next_dyn(&snapshot)               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use rolling_ta::ta::manager::{IndicatorManager, IndicatorId};
//! use rolling_ta::ta::math::SharedWindow;
//! use rolling_ta::trend::{SMA, SMAConfig};
//!
//! // Create shared window
//! let window: SharedWindow = Arc::new(RwLock::new(
//!     RollingWindow::with_timeframe(1440, 60)
//! ));
//!
//! // Create manager
//! let mut manager = IndicatorManager::new(Arc::clone(&window));
//!
//! // Register indicators
//! let sma_id = manager.register(SMA::new(SMAConfig::new(14)));
//!
//! // Run all indicators
//! manager.run();
//!
//! // Read values directly from indicator
//! if let Some(sma) = manager.get::<SMA>(sma_id) {
//!     println!("SMA: {:?}", sma.latest());
//! }
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::ta::error::{TAError, TAResult};
use crate::ta::math::SharedWindow;
use crate::ta::types::Ohlcv;
use crate::ta::Indicator;

/// Opaque handle for indicator identification.
///
/// Type-safe alternative to string names. Generated automatically
/// when registering an indicator with `IndicatorManager`.
///
/// # Example
///
/// ```ignore
/// let sma_id = manager.register(SMA::new(config));
/// // Later...
/// let sma = manager.get::<SMA>(sma_id);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndicatorId(u64);

impl IndicatorId {
    /// Generate a new unique indicator ID.
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value (for debugging/logging).
    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Internal trait combining Indicator + Any for type-erased storage with downcasting.
///
/// This trait is automatically implemented for all types that satisfy the bounds.
/// Users don't interact with this directly - use `IndicatorManager::get<T>()` instead.
pub trait DynIndicator: Send + Sync {
    /// Call `next()` on the indicator with type erasure.
    fn next_dyn(&mut self, candles: &[Ohlcv]);

    /// Call `calc()` on the indicator with type erasure.
    fn calc_dyn(&mut self, candles: &[Ohlcv]) -> TAResult<()>;

    /// Downcast to concrete type (immutable).
    fn as_any(&self) -> &dyn Any;

    /// Downcast to concrete type (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T> DynIndicator for T
where
    T: Indicator + Any + Send + Sync + 'static,
{
    fn next_dyn(&mut self, candles: &[Ohlcv]) {
        self.next(candles);
    }

    fn calc_dyn(&mut self, candles: &[Ohlcv]) -> TAResult<()> {
        self.calc(candles)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Manager for orchestrating multiple indicators with a shared window.
///
/// # Design Principles
///
/// - **Immutable Replace**: Configs are immutable. Config change = new indicator instance.
/// - **Typed Handles**: Indicators identified by `IndicatorId`, not strings.
/// - **Direct Access**: Caller retrieves indicators to read values (no god function).
/// - **Atomic Swap**: Failed replacement preserves old indicator.
///
/// # Thread Safety
///
/// The manager itself is not `Sync` (requires `&mut self` for mutations).
/// However, the `run()` method executes indicators in parallel safely
/// because each indicator is mutated independently.
pub struct IndicatorManager {
    window: SharedWindow,
    indicators: HashMap<IndicatorId, Box<dyn DynIndicator>>,
}

impl IndicatorManager {
    /// Create a new indicator manager with the given shared window.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let window: SharedWindow = Arc::new(RwLock::new(RollingWindow::new(1440)));
    /// let manager = IndicatorManager::new(Arc::clone(&window));
    /// ```
    pub fn new(window: SharedWindow) -> Self {
        Self {
            window,
            indicators: HashMap::new(),
        }
    }

    /// Register an indicator, returning a handle for later access.
    ///
    /// The indicator is stored and can be retrieved via `get()` or `get_mut()`.
    /// Use the returned `IndicatorId` to identify this indicator in future calls.
    ///
    /// # Type Parameters
    ///
    /// - `I`: Any type implementing `Indicator + 'static`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut sma = SMA::new(SMAConfig::new(14));
    /// sma.calc(&initial_candles)?;
    /// let sma_id = manager.register(sma);
    /// ```
    pub fn register<I>(&mut self, indicator: I) -> IndicatorId
    where
        I: Indicator + Any + Send + Sync + 'static,
    {
        let id = IndicatorId::new();
        self.indicators.insert(id, Box::new(indicator));
        id
    }

    /// Get an indicator by handle (immutable).
    ///
    /// Returns `None` if:
    /// - The ID doesn't exist
    /// - The type `I` doesn't match the stored indicator
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(sma) = manager.get::<SMA>(sma_id) {
    ///     println!("Latest: {:?}", sma.latest());
    ///     println!("History: {:?}", sma.history());
    /// }
    /// ```
    pub fn get<I: 'static>(&self, id: IndicatorId) -> Option<&I> {
        self.indicators.get(&id)?.as_any().downcast_ref()
    }

    /// Get an indicator by handle (mutable).
    ///
    /// Returns `None` if:
    /// - The ID doesn't exist
    /// - The type `I` doesn't match the stored indicator
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(sma) = manager.get_mut::<SMA>(sma_id) {
    ///     sma.reset();
    /// }
    /// ```
    pub fn get_mut<I: 'static>(&mut self, id: IndicatorId) -> Option<&mut I> {
        self.indicators.get_mut(&id)?.as_any_mut().downcast_mut()
    }

    /// Check if an indicator exists.
    #[inline]
    pub fn contains(&self, id: IndicatorId) -> bool {
        self.indicators.contains_key(&id)
    }

    /// Replace an indicator with a new instance.
    ///
    /// **ATOMIC**: If rehydration fails, the old indicator is preserved.
    ///
    /// This is the recommended way to change indicator configuration:
    /// 1. Create new indicator with new config
    /// 2. Call `replace()` - it will rehydrate from SharedWindow
    /// 3. On success, old indicator is replaced
    /// 4. On failure, old indicator remains (error returned)
    ///
    /// # Errors
    ///
    /// - `TAError::InvalidId` if the ID doesn't exist
    /// - Any error from `calc()` during rehydration
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Change SMA period from 14 to 20
    /// manager.replace(sma_id, SMA::new(SMAConfig::new(20)))?;
    /// ```
    pub fn replace<I>(&mut self, id: IndicatorId, mut new_indicator: I) -> TAResult<()>
    where
        I: Indicator + Any + Send + Sync + 'static,
    {
        if !self.indicators.contains_key(&id) {
            return Err(TAError::InvalidId);
        }

        // Rehydrate BEFORE replacing (atomic: old preserved on failure)
        let snapshot = self.window.read().unwrap().snapshot();
        new_indicator.calc(&snapshot)?;

        // Only mutate after success
        self.indicators.insert(id, Box::new(new_indicator));
        Ok(())
    }

    /// Run all indicators in parallel.
    ///
    /// Takes a snapshot from the SharedWindow and calls `next()` on each
    /// indicator concurrently via Rayon. Results are stored in each
    /// indicator's internal state - use `get()` to read values afterward.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Push new candle to window
    /// window.write().unwrap().push(candle);
    ///
    /// // Run all indicators
    /// manager.run();
    ///
    /// // Read results
    /// let sma_value = manager.get::<SMA>(sma_id)?.latest();
    /// ```
    pub fn run(&mut self) {
        let snapshot = self.window.read().unwrap().snapshot();

        // Parallel execution via Rayon
        // SAFETY: Each indicator is mutated independently (no shared state between them)
        self.indicators
            .par_iter_mut()
            .for_each(|(_, ind)| {
                ind.next_dyn(&snapshot);
            });
    }

    /// Run a single indicator by ID.
    ///
    /// Returns `false` if the ID doesn't exist.
    pub fn run_one(&mut self, id: IndicatorId) -> bool {
        let snapshot = self.window.read().unwrap().snapshot();

        if let Some(ind) = self.indicators.get_mut(&id) {
            ind.next_dyn(&snapshot);
            true
        } else {
            false
        }
    }

    /// Remove an indicator.
    ///
    /// Returns `true` if the indicator was removed, `false` if ID didn't exist.
    pub fn remove(&mut self, id: IndicatorId) -> bool {
        self.indicators.remove(&id).is_some()
    }

    /// Number of registered indicators.
    #[inline]
    pub fn len(&self) -> usize {
        self.indicators.len()
    }

    /// Check if no indicators are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indicators.is_empty()
    }

    /// Get all indicator IDs.
    pub fn ids(&self) -> impl Iterator<Item = IndicatorId> + '_ {
        self.indicators.keys().copied()
    }

    /// Get a reference to the shared window.
    #[inline]
    pub fn window(&self) -> &SharedWindow {
        &self.window
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ta::config::SMAConfig;
    use crate::ta::math::RollingWindow;
    use crate::ta::trend::SMA;
    use crate::ta::HistoricalIndicator;
    use std::sync::{Arc, RwLock};

    fn create_test_window(candles: &[Ohlcv]) -> SharedWindow {
        let mut window = RollingWindow::new(100);
        for candle in candles {
            window.push(*candle);
        }
        Arc::new(RwLock::new(window))
    }

    fn generate_candles(n: usize) -> Vec<Ohlcv> {
        (0..n)
            .map(|i| Ohlcv::from_close(100.0 + i as f64))
            .collect()
    }

    #[test]
    fn register_and_get() {
        let candles = generate_candles(20);
        let window = create_test_window(&candles);
        let mut manager = IndicatorManager::new(window);

        let mut sma = SMA::new(SMAConfig::new(5));
        sma.calc(&candles).unwrap();
        let sma_id = manager.register(sma);

        assert!(manager.contains(sma_id));
        assert_eq!(manager.len(), 1);

        // Get by correct type
        let retrieved = manager.get::<SMA>(sma_id);
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().latest().is_some());

        // Get by wrong type should fail
        let wrong: Option<&crate::ta::trend::EMA> = manager.get(sma_id);
        assert!(wrong.is_none());
    }

    #[test]
    fn run_updates_indicators() {
        let candles = generate_candles(20);
        let window = create_test_window(&candles);
        let mut manager = IndicatorManager::new(Arc::clone(&window));

        // Register fresh (uncalculated) indicator
        let sma = SMA::new(SMAConfig::new(5));
        let sma_id = manager.register(sma);

        // Before run: no latest value
        assert!(manager.get::<SMA>(sma_id).unwrap().latest().is_none());

        // Run should call next() which produces a value
        manager.run();

        // After run: should have latest value
        let sma = manager.get::<SMA>(sma_id).unwrap();
        assert!(sma.latest().is_some());
    }

    #[test]
    fn replace_atomic_on_success() {
        let candles = generate_candles(30);
        let window = create_test_window(&candles);
        let mut manager = IndicatorManager::new(window);

        // Register SMA with period 5
        let mut sma5 = SMA::new(SMAConfig::new(5));
        sma5.calc(&candles).unwrap();
        let sma_id = manager.register(sma5);

        let old_latest = manager.get::<SMA>(sma_id).unwrap().latest();

        // Replace with period 10
        let result = manager.replace(sma_id, SMA::new(SMAConfig::new(10)));
        assert!(result.is_ok());

        // New indicator should be rehydrated with different value
        let new_sma = manager.get::<SMA>(sma_id).unwrap();
        assert!(new_sma.latest().is_some());
        // Period 10 SMA will have different value than period 5
        assert_ne!(new_sma.latest(), old_latest);
    }

    #[test]
    fn replace_invalid_id() {
        let window = create_test_window(&generate_candles(10));
        let mut manager = IndicatorManager::new(window);

        let fake_id = IndicatorId::new();
        let result = manager.replace(fake_id, SMA::new(SMAConfig::new(5)));

        assert!(matches!(result, Err(TAError::InvalidId)));
    }

    #[test]
    fn remove_indicator() {
        let candles = generate_candles(20);
        let window = create_test_window(&candles);
        let mut manager = IndicatorManager::new(window);

        let sma_id = manager.register(SMA::new(SMAConfig::new(5)));
        assert_eq!(manager.len(), 1);

        let removed = manager.remove(sma_id);
        assert!(removed);
        assert_eq!(manager.len(), 0);
        assert!(!manager.contains(sma_id));

        // Remove again should return false
        let removed_again = manager.remove(sma_id);
        assert!(!removed_again);
    }

    #[test]
    fn multiple_indicators() {
        let candles = generate_candles(30);
        let window = create_test_window(&candles);
        let mut manager = IndicatorManager::new(window);

        let sma5_id = manager.register(SMA::new(SMAConfig::new(5)));
        let sma10_id = manager.register(SMA::new(SMAConfig::new(10)));
        let sma20_id = manager.register(SMA::new(SMAConfig::new(20)));

        assert_eq!(manager.len(), 3);

        // Run all in parallel
        manager.run();

        // All should have values
        assert!(manager.get::<SMA>(sma5_id).unwrap().latest().is_some());
        assert!(manager.get::<SMA>(sma10_id).unwrap().latest().is_some());
        assert!(manager.get::<SMA>(sma20_id).unwrap().latest().is_some());
    }

    #[test]
    fn get_mut_and_reset() {
        let candles = generate_candles(20);
        let window = create_test_window(&candles);
        let mut manager = IndicatorManager::new(window);

        let mut sma = SMA::new(SMAConfig::new(5));
        sma.calc(&candles).unwrap();
        let sma_id = manager.register(sma);

        // Has history
        assert!(!manager.get::<SMA>(sma_id).unwrap().history().is_empty());

        // Reset via mutable access
        manager.get_mut::<SMA>(sma_id).unwrap().reset();

        // History should be cleared
        assert!(manager.get::<SMA>(sma_id).unwrap().history().is_empty());
    }

    #[test]
    fn indicator_id_uniqueness() {
        let id1 = IndicatorId::new();
        let id2 = IndicatorId::new();
        let id3 = IndicatorId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn run_one() {
        let candles = generate_candles(20);
        let window = create_test_window(&candles);
        let mut manager = IndicatorManager::new(window);

        let sma_id = manager.register(SMA::new(SMAConfig::new(5)));

        assert!(manager.run_one(sma_id));
        assert!(manager.get::<SMA>(sma_id).unwrap().latest().is_some());

        // Invalid ID
        let fake_id = IndicatorId::new();
        assert!(!manager.run_one(fake_id));
    }

    #[test]
    fn ids_iterator() {
        let window = create_test_window(&generate_candles(10));
        let mut manager = IndicatorManager::new(window);

        let id1 = manager.register(SMA::new(SMAConfig::new(5)));
        let id2 = manager.register(SMA::new(SMAConfig::new(10)));

        let ids: Vec<_> = manager.ids().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }
}