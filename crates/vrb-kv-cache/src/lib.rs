#![forbid(unsafe_code)]

//! Capacity-bounded KV-cache metadata layered on durable residency keys.
//!
//! This crate never stores or retains raw OS/GPU handles. The actual allocations
//! remain owned by a residency provider.

use std::collections::BTreeMap;

use thiserror::Error;
use vrb_residency::ResidencyKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvCacheKey {
    pub sequence_id: u64,
    pub layer: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBlock {
    pub token_start: u64,
    pub token_count: u32,
    pub key_resource: ResidencyKey,
    pub value_resource: ResidencyKey,
    /// Combined bytes owned by the K and V resources represented by this block.
    pub bytes: u64,
}

impl KvBlock {
    pub fn validate(&self) -> Result<(), KvCacheError> {
        if self.token_count == 0 || self.bytes == 0 {
            return Err(KvCacheError::InvalidBlock);
        }
        self.token_end()?;
        Ok(())
    }

    pub fn token_end(&self) -> Result<u64, KvCacheError> {
        self.token_start
            .checked_add(u64::from(self.token_count))
            .ok_or(KvCacheError::TokenRangeOverflow)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KvEntry {
    blocks: Vec<KvBlock>,
    bytes: u64,
    last_touch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheView {
    pub key: KvCacheKey,
    pub blocks: Vec<KvBlock>,
    pub bytes: u64,
    pub tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvCacheStats {
    pub entries: usize,
    pub blocks: usize,
    pub bytes: u64,
    pub evictions: u64,
}

pub struct KvCache {
    capacity_bytes: u64,
    entries: BTreeMap<KvCacheKey, KvEntry>,
    bytes: u64,
    clock: u64,
    evictions: u64,
}

impl KvCache {
    pub fn new(capacity_bytes: u64) -> Result<Self, KvCacheError> {
        if capacity_bytes == 0 {
            return Err(KvCacheError::ZeroCapacity);
        }
        Ok(Self {
            capacity_bytes,
            entries: BTreeMap::new(),
            bytes: 0,
            clock: 0,
            evictions: 0,
        })
    }

    pub fn append(&mut self, key: KvCacheKey, block: KvBlock) -> Result<(), KvCacheError> {
        block.validate()?;
        if block.bytes > self.capacity_bytes {
            return Err(KvCacheError::CapacityExceeded {
                requested: block.bytes,
                capacity: self.capacity_bytes,
            });
        }

        if let Some(entry) = self.entries.get(&key) {
            let expected_start = entry
                .blocks
                .last()
                .ok_or(KvCacheError::InvalidBlock)?
                .token_end()?;
            if block.token_start != expected_start {
                return Err(KvCacheError::NonContiguous {
                    expected_start,
                    actual_start: block.token_start,
                });
            }
        } else if block.token_start != 0 {
            return Err(KvCacheError::NonContiguous {
                expected_start: 0,
                actual_start: block.token_start,
            });
        }

        let current_entry_bytes = self.entries.get(&key).map_or(0, |entry| entry.bytes);
        let new_entry_bytes = current_entry_bytes
            .checked_add(block.bytes)
            .ok_or(KvCacheError::SizeOverflow)?;
        if new_entry_bytes > self.capacity_bytes {
            return Err(KvCacheError::CapacityExceeded {
                requested: new_entry_bytes,
                capacity: self.capacity_bytes,
            });
        }

        self.evict_for(key, block.bytes)?;
        self.clock = self.clock.wrapping_add(1).max(1);
        let touch = self.clock;
        let entry = self.entries.entry(key).or_insert_with(|| KvEntry {
            blocks: Vec::new(),
            bytes: 0,
            last_touch: touch,
        });
        entry.bytes = entry
            .bytes
            .checked_add(block.bytes)
            .ok_or(KvCacheError::SizeOverflow)?;
        entry.blocks.push(block);
        entry.last_touch = touch;
        self.bytes = self
            .bytes
            .checked_add(entry.blocks.last().expect("block was just pushed").bytes)
            .ok_or(KvCacheError::SizeOverflow)?;
        Ok(())
    }

    pub fn view(&mut self, key: KvCacheKey) -> Result<Option<KvCacheView>, KvCacheError> {
        self.clock = self.clock.wrapping_add(1).max(1);
        let touch = self.clock;
        let Some(entry) = self.entries.get_mut(&key) else {
            return Ok(None);
        };
        entry.last_touch = touch;
        let tokens = entry
            .blocks
            .last()
            .map(KvBlock::token_end)
            .transpose()?
            .unwrap_or(0);
        Ok(Some(KvCacheView {
            key,
            blocks: entry.blocks.clone(),
            bytes: entry.bytes,
            tokens,
        }))
    }

    pub fn remove(&mut self, key: KvCacheKey) -> bool {
        if let Some(entry) = self.entries.remove(&key) {
            self.bytes = self.bytes.saturating_sub(entry.bytes);
            return true;
        }
        false
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    #[must_use]
    pub fn stats(&self) -> KvCacheStats {
        KvCacheStats {
            entries: self.entries.len(),
            blocks: self.entries.values().map(|entry| entry.blocks.len()).sum(),
            bytes: self.bytes,
            evictions: self.evictions,
        }
    }

    fn evict_for(&mut self, protected: KvCacheKey, additional: u64) -> Result<(), KvCacheError> {
        while self
            .bytes
            .checked_add(additional)
            .is_none_or(|total| total > self.capacity_bytes)
        {
            let victim = self
                .entries
                .iter()
                .filter(|(key, _)| **key != protected)
                .min_by_key(|(_, entry)| entry.last_touch)
                .map(|(key, _)| *key)
                .ok_or(KvCacheError::CapacityExceeded {
                    requested: self
                        .bytes
                        .checked_add(additional)
                        .unwrap_or(u64::MAX),
                    capacity: self.capacity_bytes,
                })?;
            let removed = self
                .entries
                .remove(&victim)
                .expect("selected KV-cache victim must exist");
            self.bytes = self.bytes.saturating_sub(removed.bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    #[error("KV-cache capacity must be non-zero")]
    ZeroCapacity,
    #[error("KV block must have non-zero token count and byte size")]
    InvalidBlock,
    #[error("KV token range overflow")]
    TokenRangeOverflow,
    #[error("KV cache append is not contiguous: expected token {expected_start}, got {actual_start}")]
    NonContiguous {
        expected_start: u64,
        actual_start: u64,
    },
    #[error("KV-cache size arithmetic overflow")]
    SizeOverflow,
    #[error("KV-cache capacity exceeded: requested {requested}, capacity {capacity}")]
    CapacityExceeded { requested: u64, capacity: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(start: u64, count: u32, bytes: u64, suffix: &str) -> KvBlock {
        KvBlock {
            token_start: start,
            token_count: count,
            key_resource: ResidencyKey::new(format!("k-{suffix}")).unwrap(),
            value_resource: ResidencyKey::new(format!("v-{suffix}")).unwrap(),
            bytes,
        }
    }

    #[test]
    fn appends_must_be_contiguous() {
        let key = KvCacheKey {
            sequence_id: 1,
            layer: 0,
        };
        let mut cache = KvCache::new(1024).unwrap();
        cache.append(key, block(0, 4, 100, "a")).unwrap();
        assert!(matches!(
            cache.append(key, block(5, 1, 10, "b")),
            Err(KvCacheError::NonContiguous {
                expected_start: 4,
                actual_start: 5
            })
        ));
    }

    #[test]
    fn lru_eviction_preserves_recent_entry() {
        let a = KvCacheKey {
            sequence_id: 1,
            layer: 0,
        };
        let b = KvCacheKey {
            sequence_id: 2,
            layer: 0,
        };
        let c = KvCacheKey {
            sequence_id: 3,
            layer: 0,
        };
        let mut cache = KvCache::new(100).unwrap();
        cache.append(a, block(0, 1, 40, "a")).unwrap();
        cache.append(b, block(0, 1, 40, "b")).unwrap();
        cache.view(a).unwrap();
        cache.append(c, block(0, 1, 40, "c")).unwrap();

        assert!(cache.view(a).unwrap().is_some());
        assert!(cache.view(b).unwrap().is_none());
        assert!(cache.view(c).unwrap().is_some());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn active_entry_is_not_partially_evicted() {
        let key = KvCacheKey {
            sequence_id: 1,
            layer: 0,
        };
        let mut cache = KvCache::new(100).unwrap();
        cache.append(key, block(0, 1, 60, "a")).unwrap();
        assert!(matches!(
            cache.append(key, block(1, 1, 60, "b")),
            Err(KvCacheError::CapacityExceeded { .. })
        ));
        assert_eq!(cache.view(key).unwrap().unwrap().tokens, 1);
    }
}
