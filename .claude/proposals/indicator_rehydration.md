# Indicator Rehydration Design Proposal

## Problem Statement

Indicator configuration may change at runtime (e.g., SMA period 14 → 20). When this happens:
1. Internal state (history, prev_ema, avg_gain, etc.) becomes invalid
2. Need to "rehydrate" - recalculate from SharedWindow snapshot

## Chosen Approach: Immutable Replace

**Rationale:** Lowest complexity, excellent SOLID adherence.

### Core Principle

> Configs are immutable. Config change = new indicator instance.

```
Config change detected
    → Create new indicator with new config
    → Rehydrate via calc(&snapshot)
    → Replace old indicator in Manager
```

## Design

### Design Decisions (from Q&A)

| Question | Decision | Implication |
|----------|----------|-------------|
| Identification | Typed handles | `IndicatorId` newtype, compile-time safety |
| Output | Get indicators directly | No aggregation, caller reads from indicator |
| Error handling | Atomic swap | Old indicator preserved on failure |

### IndicatorId

```rust
/// Opaque handle for indicator identification.
/// Type-safe alternative to string names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndicatorId(u64);

impl IndicatorId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}
```

### Type Erasure Solution

To store heterogeneous indicators while allowing caller to retrieve concrete types:

```rust
use std::any::Any;

/// Internal trait combining Indicator + Any for downcasting.
pub(crate) trait DynIndicator: Send + Sync {
    fn next_dyn(&mut self, candles: &[Ohlcv]);
    fn calc_dyn(&mut self, candles: &[Ohlcv]) -> TAResult<()>;
    fn as_any(&self) -> &dyn Any;
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
```

### IndicatorManager API

```rust
pub struct IndicatorManager {
    window: SharedWindow,
    indicators: HashMap<IndicatorId, Box<dyn DynIndicator>>,
}

impl IndicatorManager {
    pub fn new(window: SharedWindow) -> Self {
        Self {
            window,
            indicators: HashMap::new(),
        }
    }

    /// Register an indicator, returns handle for later access.
    pub fn register<I>(&mut self, indicator: I) -> IndicatorId
    where
        I: Indicator + Any + Send + Sync + 'static,
    {
        let id = IndicatorId::new();
        self.indicators.insert(id, Box::new(indicator));
        id
    }

    /// Get indicator by handle (immutable).
    /// Caller specifies concrete type for downcasting.
    pub fn get<I: 'static>(&self, id: IndicatorId) -> Option<&I> {
        self.indicators.get(&id)?.as_any().downcast_ref()
    }

    /// Get indicator by handle (mutable).
    pub fn get_mut<I: 'static>(&mut self, id: IndicatorId) -> Option<&mut I> {
        self.indicators.get_mut(&id)?.as_any_mut().downcast_mut()
    }

    /// Replace indicator with new instance.
    /// ATOMIC: Old indicator preserved if rehydration fails.
    pub fn replace<I>(&mut self, id: IndicatorId, mut new_indicator: I) -> TAResult<()>
    where
        I: Indicator + Any + Send + Sync + 'static,
    {
        if !self.indicators.contains_key(&id) {
            return Err(TAError::InvalidId);
        }

        // Rehydrate BEFORE replacing
        let snapshot = self.window.read().unwrap().snapshot();
        new_indicator.calc(&snapshot)?;  // If fails, old indicator untouched

        // Only mutate after success
        self.indicators.insert(id, Box::new(new_indicator));
        Ok(())
    }

    /// Run all indicators in parallel.
    /// Updates internal state; caller reads values via get().
    pub fn run(&mut self) {
        let snapshot = self.window.read().unwrap().snapshot();

        self.indicators
            .par_iter_mut()
            .for_each(|(_, ind)| {
                ind.next_dyn(&snapshot);
            });
    }

    /// Remove an indicator.
    pub fn remove(&mut self, id: IndicatorId) -> bool {
        self.indicators.remove(&id).is_some()
    }

    /// Number of registered indicators.
    pub fn len(&self) -> usize {
        self.indicators.len()
    }
}
```

### Usage Pattern

```rust
// Setup
let window: SharedWindow = Arc::new(RwLock::new(RollingWindow::new(1440)));
let mut manager = IndicatorManager::new(Arc::clone(&window));

// Register indicators (keeps handle)
let mut sma = SMA::new(SMAConfig::new(14));
sma.calc(&initial_candles)?;
let sma_id = manager.register(sma);

let mut ema = EMA::new(EMAConfig::new(14));
ema.calc(&initial_candles)?;
let ema_id = manager.register(ema);

// Streaming loop
loop {
    // Push new candle to window (external)
    window.write().unwrap().push(new_candle);

    // Run all indicators in parallel
    manager.run();

    // Read values directly from indicators
    if let Some(sma) = manager.get::<SMA>(sma_id) {
        println!("SMA: {:?}", sma.latest());
        println!("SMA history: {:?}", sma.history());
    }

    if let Some(ema) = manager.get::<EMA>(ema_id) {
        println!("EMA: {:?}", ema.latest());
    }
}

// Config change: replace SMA with period 20
// Old SMA preserved if this fails
manager.replace(sma_id, SMA::new(SMAConfig::new(20)))?;
```

## SOLID Compliance

| Principle | How It's Satisfied |
|-----------|-------------------|
| **SRP** | Indicator: calculation only. Manager: lifecycle only. |
| **OCP** | New indicator types require no changes to Manager |
| **LSP** | All indicators substitutable via trait |
| **ISP** | No trait changes, minimal interface |
| **DIP** | Manager depends on Indicator trait abstraction |

## Trade-offs Acknowledged

| Trade-off | Mitigation |
|-----------|------------|
| History lost on replace | Expected behavior - new config means new history |
| Allocation on replace | Negligible for infrequent config changes |
| Type erasure for storage | `DynIndicator` trait + `Any` downcasting |
| Caller must know concrete type | Caller registered it, so they know the type |
| Downcast can fail | Returns `Option`, caller handles `None` |

## What This Approach Avoids

- No `Stale` state in `IndicatorState`
- No `invalidate()` or `is_stale()` methods
- No per-tick staleness checks
- No automatic detection logic
- No config hash tracking

## Open Questions

> **Instructions:** Answer these questions by editing this section.

### Q1: Indicator Identification

How should indicators be identified in the Manager?

- [ ] **A) String names** - `"sma_14"`, `"ema_fast"` (flexible, familiar)
- [ ] **B) Typed handles** - `IndicatorId(uuid)` (type-safe, opaque)
- [ ] **C) Index-based** - Position in Vec (simple, but fragile on removal)

**Your choice:** Typed handles

---

### Q2: Output Type Flexibility

Should Manager support heterogeneous output types?

- [ ] **A) Single type** - `IndicatorManager<f64>` - all indicators return f64
- [ ] **B) Enum output** - `IndicatorOutput { Scalar(f64), Tuple(f64, f64), ... }`
- [ ] **C) Type-erased** - `Box<dyn Any>` (flexible but loses type safety)

**Your choice:** None, instead of outputting values itself, the IndicatorManager should provide the functionality to get initialized indicators. If the indicator is a HistoricalIndicator we will be able to read its calculation history. We don't want to try to create a god function for reading the values of all indicators, there could be alot.

---

### Q3: Error Handling on Replace

If rehydration fails during `replace_rehydrated()`, should we:

- [ ] **A) Keep old indicator** - Atomic swap, only replace on success
- [ ] **B) Remove indicator** - Fail-fast, indicator gone until re-registered
- [ ] **C) Return error, no change** - Current behavior, caller decides

**Your choice:** Keep old indicator

---

## Next Steps

Once questions are answered:

1. Implement `IndicatorManager` in `src/ta/manager.rs`
2. Add `SharedWindow` type alias to `src/ta/math/rolling.rs`
3. Add `snapshot()` method to `RollingWindow`
4. Integration tests for replace + rehydration
5. Benchmark parallel vs sequential execution
