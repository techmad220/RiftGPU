#![forbid(unsafe_code)]

//! Durable GPU-resource residency contracts.
//!
//! The shared-operator ABI intentionally exposes borrowed native handles that
//! cannot be retained beyond a callback. This crate models the longer-lived
//! owner/provider boundary needed to keep imported GPU resources resident
//! safely across repeated operations without violating that ABI contract.

use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
    sync::{Arc, Mutex},
};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidencyKey(String);

impl ResidencyKey {
    pub fn new(value: impl Into<String>) -> Result<Self, ResidencyError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(ResidencyError::InvalidKey);
        }
        Ok(Self(trimmed.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ResidencyKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencySpec {
    pub key: ResidencyKey,
    /// Resource generation. Bump whenever the owning allocation is replaced.
    pub generation: u64,
    pub device_index: i32,
    pub bytes: u64,
}

impl ResidencySpec {
    pub fn validate(&self) -> Result<(), ResidencyError> {
        if self.device_index < 0 || self.bytes == 0 {
            return Err(ResidencyError::InvalidSpec {
                key: self.key.clone(),
                device_index: self.device_index,
                bytes: self.bytes,
            });
        }
        Ok(())
    }
}

/// Provider-owned durable allocation/import.
///
/// Implementations may own native Vulkan/HIP objects internally. Those objects
/// are never extracted as retained borrowed handles through this crate.
pub trait ResidentAllocation: Send + Sync {
    fn spec(&self) -> &ResidencySpec;
}

/// Opens a durable resource whose lifetime is explicitly owned by the returned
/// allocation object.
pub trait ResidencyProvider: Send + Sync {
    fn open(&self, spec: &ResidencySpec) -> Result<Arc<dyn ResidentAllocation>, ResidencyError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencyStats {
    pub entries: usize,
    pub resident_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct CacheEntry {
    allocation: Arc<dyn ResidentAllocation>,
    last_touch: u64,
}

#[derive(Default)]
struct CacheState {
    entries: BTreeMap<ResidencyKey, CacheEntry>,
    resident_bytes: u64,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

pub struct ResidencyCache {
    capacity_bytes: u64,
    state: Mutex<CacheState>,
}

impl ResidencyCache {
    pub fn new(capacity_bytes: u64) -> Result<Self, ResidencyError> {
        if capacity_bytes == 0 {
            return Err(ResidencyError::ZeroCapacity);
        }
        Ok(Self {
            capacity_bytes,
            state: Mutex::new(CacheState::default()),
        })
    }

    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub fn acquire(
        &self,
        provider: &dyn ResidencyProvider,
        spec: &ResidencySpec,
    ) -> Result<Arc<dyn ResidentAllocation>, ResidencyError> {
        spec.validate()?;
        if spec.bytes > self.capacity_bytes {
            return Err(ResidencyError::CapacityExceeded {
                requested: spec.bytes,
                capacity: self.capacity_bytes,
            });
        }

        let mut state = self.state.lock().map_err(|_| ResidencyError::Poisoned)?;
        state.clock = state.clock.wrapping_add(1).max(1);
        let touch = state.clock;

        if let Some(entry) = state.entries.get_mut(&spec.key) {
            if entry.allocation.spec() == spec {
                entry.last_touch = touch;
                let allocation = Arc::clone(&entry.allocation);
                state.hits = state.hits.saturating_add(1);
                return Ok(allocation);
            }
        }

        if let Some(stale) = state.entries.remove(&spec.key) {
            state.resident_bytes = state
                .resident_bytes
                .saturating_sub(stale.allocation.spec().bytes);
            state.evictions = state.evictions.saturating_add(1);
        }

        state.misses = state.misses.saturating_add(1);
        while state
            .resident_bytes
            .checked_add(spec.bytes)
            .is_none_or(|total| total > self.capacity_bytes)
        {
            let victim = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_touch)
                .map(|(key, _)| key.clone())
                .ok_or(ResidencyError::CapacityExceeded {
                    requested: spec.bytes,
                    capacity: self.capacity_bytes,
                })?;
            let removed = state
                .entries
                .remove(&victim)
                .expect("selected residency victim must exist");
            state.resident_bytes = state
                .resident_bytes
                .saturating_sub(removed.allocation.spec().bytes);
            state.evictions = state.evictions.saturating_add(1);
        }

        let allocation = provider.open(spec)?;
        if allocation.spec() != spec {
            return Err(ResidencyError::ProviderContract {
                requested: spec.clone(),
                returned: allocation.spec().clone(),
            });
        }

        state.resident_bytes = state.resident_bytes.checked_add(spec.bytes).ok_or(
            ResidencyError::CapacityExceeded {
                requested: spec.bytes,
                capacity: self.capacity_bytes,
            },
        )?;
        state.entries.insert(
            spec.key.clone(),
            CacheEntry {
                allocation: Arc::clone(&allocation),
                last_touch: touch,
            },
        );
        Ok(allocation)
    }

    pub fn invalidate(&self, key: &ResidencyKey) -> Result<bool, ResidencyError> {
        let mut state = self.state.lock().map_err(|_| ResidencyError::Poisoned)?;
        if let Some(removed) = state.entries.remove(key) {
            state.resident_bytes = state
                .resident_bytes
                .saturating_sub(removed.allocation.spec().bytes);
            state.evictions = state.evictions.saturating_add(1);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn clear(&self) -> Result<(), ResidencyError> {
        let mut state = self.state.lock().map_err(|_| ResidencyError::Poisoned)?;
        state.evictions = state
            .evictions
            .saturating_add(u64::try_from(state.entries.len()).unwrap_or(u64::MAX));
        state.entries.clear();
        state.resident_bytes = 0;
        Ok(())
    }

    pub fn stats(&self) -> Result<ResidencyStats, ResidencyError> {
        let state = self.state.lock().map_err(|_| ResidencyError::Poisoned)?;
        Ok(ResidencyStats {
            entries: state.entries.len(),
            resident_bytes: state.resident_bytes,
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ResidencyError {
    #[error("residency key cannot be empty")]
    InvalidKey,
    #[error("invalid residency spec for '{key}': device_index={device_index}, bytes={bytes}")]
    InvalidSpec {
        key: ResidencyKey,
        device_index: i32,
        bytes: u64,
    },
    #[error("residency cache capacity must be non-zero")]
    ZeroCapacity,
    #[error("residency capacity exceeded: requested {requested} bytes, capacity {capacity} bytes")]
    CapacityExceeded { requested: u64, capacity: u64 },
    #[error("residency provider failed: {0}")]
    Provider(String),
    #[error(
        "residency provider contract violation: requested {requested:?}, returned {returned:?}"
    )]
    ProviderContract {
        requested: ResidencySpec,
        returned: ResidencySpec,
    },
    #[error("residency cache lock is poisoned")]
    Poisoned,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeAllocation {
        spec: ResidencySpec,
    }

    impl ResidentAllocation for FakeAllocation {
        fn spec(&self) -> &ResidencySpec {
            &self.spec
        }
    }

    struct FakeProvider {
        opens: AtomicUsize,
    }

    impl FakeProvider {
        fn new() -> Self {
            Self {
                opens: AtomicUsize::new(0),
            }
        }
    }

    impl ResidencyProvider for FakeProvider {
        fn open(
            &self,
            spec: &ResidencySpec,
        ) -> Result<Arc<dyn ResidentAllocation>, ResidencyError> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakeAllocation { spec: spec.clone() }))
        }
    }

    fn spec(key: &str, generation: u64, bytes: u64) -> ResidencySpec {
        ResidencySpec {
            key: ResidencyKey::new(key).unwrap(),
            generation,
            device_index: 0,
            bytes,
        }
    }

    #[test]
    fn exact_generation_is_reused() {
        let provider = FakeProvider::new();
        let cache = ResidencyCache::new(1024).unwrap();
        let first = cache.acquire(&provider, &spec("weights", 1, 512)).unwrap();
        let second = cache.acquire(&provider, &spec("weights", 1, 512)).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(provider.opens.load(Ordering::SeqCst), 1);
        assert_eq!(
            cache.stats().unwrap(),
            ResidencyStats {
                entries: 1,
                resident_bytes: 512,
                hits: 1,
                misses: 1,
                evictions: 0,
            }
        );
    }

    #[test]
    fn generation_change_reopens_without_invalidating_active_lease() {
        let provider = FakeProvider::new();
        let cache = ResidencyCache::new(1024).unwrap();
        let old = cache.acquire(&provider, &spec("weights", 1, 512)).unwrap();
        let new = cache.acquire(&provider, &spec("weights", 2, 512)).unwrap();

        assert!(!Arc::ptr_eq(&old, &new));
        assert_eq!(old.spec().generation, 1);
        assert_eq!(new.spec().generation, 2);
        assert_eq!(provider.opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn least_recently_used_entry_is_evicted() {
        let provider = FakeProvider::new();
        let cache = ResidencyCache::new(100).unwrap();

        let _ = cache.acquire(&provider, &spec("a", 1, 40)).unwrap();
        let _ = cache.acquire(&provider, &spec("b", 1, 40)).unwrap();
        let _ = cache.acquire(&provider, &spec("a", 1, 40)).unwrap();
        let _ = cache.acquire(&provider, &spec("c", 1, 40)).unwrap();
        let _ = cache.acquire(&provider, &spec("b", 1, 40)).unwrap();

        assert_eq!(provider.opens.load(Ordering::SeqCst), 4);
        let stats = cache.stats().unwrap();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.resident_bytes, 80);
        assert!(stats.evictions >= 2);
    }
}
