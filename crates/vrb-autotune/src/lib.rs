#![forbid(unsafe_code)]

//! Environment- and shape-aware benchmark persistence for routing.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vrb_core::{
    BackendId, DataType, OperationKind, PerformanceRecord, PerformanceTable,
};

pub const AUTOTUNE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentFingerprint {
    pub gpu_arch: String,
    pub driver_version: String,
    pub runtime_version: String,
    pub plugin_version: String,
}

impl EnvironmentFingerprint {
    pub fn validate(&self) -> Result<(), AutotuneError> {
        for (name, value) in [
            ("gpu_arch", self.gpu_arch.as_str()),
            ("driver_version", self.driver_version.as_str()),
            ("runtime_version", self.runtime_version.as_str()),
            ("plugin_version", self.plugin_version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(AutotuneError::InvalidFingerprint(name));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadShape {
    pub dims: Vec<u64>,
    pub alignment: u32,
}

impl WorkloadShape {
    pub fn validate(&self) -> Result<(), AutotuneError> {
        if self.dims.is_empty() || self.dims.contains(&0) {
            return Err(AutotuneError::InvalidShape);
        }
        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(AutotuneError::InvalidAlignment(self.alignment));
        }
        self.dims.iter().try_fold(1_u64, |acc, dim| {
            acc.checked_mul(*dim).ok_or(AutotuneError::ShapeOverflow)
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningKey {
    pub environment: EnvironmentFingerprint,
    pub backend: BackendId,
    pub operation: OperationKind,
    pub data_type: DataType,
    pub shape: WorkloadShape,
}

impl TuningKey {
    pub fn validate(&self) -> Result<(), AutotuneError> {
        self.environment.validate()?;
        self.shape.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningRecord {
    pub key: TuningKey,
    pub median_microseconds: f64,
    pub samples: u32,
}

impl TuningRecord {
    pub fn validate(&self) -> Result<(), AutotuneError> {
        self.key.validate()?;
        if !self.median_microseconds.is_finite() || self.median_microseconds < 0.0 {
            return Err(AutotuneError::InvalidLatency(self.median_microseconds));
        }
        if self.samples == 0 {
            return Err(AutotuneError::ZeroSamples);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutotuneDatabase {
    pub schema_version: u32,
    pub records: Vec<TuningRecord>,
}

impl Default for AutotuneDatabase {
    fn default() -> Self {
        Self {
            schema_version: AUTOTUNE_SCHEMA_VERSION,
            records: Vec::new(),
        }
    }
}

impl AutotuneDatabase {
    pub fn record(&mut self, record: TuningRecord) -> Result<(), AutotuneError> {
        record.validate()?;
        if let Some(existing) = self
            .records
            .iter_mut()
            .find(|existing| existing.key == record.key)
        {
            *existing = record;
        } else {
            self.records.push(record);
        }
        Ok(())
    }

    #[must_use]
    pub fn lookup(&self, key: &TuningKey) -> Option<&TuningRecord> {
        self.records.iter().find(|record| &record.key == key)
    }

    pub fn performance_table_for(
        &self,
        environment: &EnvironmentFingerprint,
        shape: &WorkloadShape,
    ) -> Result<PerformanceTable, AutotuneError> {
        environment.validate()?;
        shape.validate()?;
        let mut table = PerformanceTable::default();
        for record in self.records.iter().filter(|record| {
            &record.key.environment == environment && &record.key.shape == shape
        }) {
            record.validate()?;
            table.record(PerformanceRecord {
                backend: record.key.backend.clone(),
                operation: record.key.operation,
                data_type: record.key.data_type,
                median_microseconds: record.median_microseconds,
                samples: record.samples,
            });
        }
        Ok(table)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, AutotuneError> {
        let bytes = fs::read(path.as_ref()).map_err(AutotuneError::Io)?;
        let database: Self = serde_json::from_slice(&bytes).map_err(AutotuneError::Json)?;
        if database.schema_version != AUTOTUNE_SCHEMA_VERSION {
            return Err(AutotuneError::UnsupportedSchema(
                database.schema_version,
            ));
        }
        for record in &database.records {
            record.validate()?;
        }
        Ok(database)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), AutotuneError> {
        if self.schema_version != AUTOTUNE_SCHEMA_VERSION {
            return Err(AutotuneError::UnsupportedSchema(self.schema_version));
        }
        for record in &self.records {
            record.validate()?;
        }

        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(AutotuneError::Io)?;
            }
        }
        let bytes = serde_json::to_vec_pretty(self).map_err(AutotuneError::Json)?;
        let temporary = temporary_path(path);
        fs::write(&temporary, bytes).map_err(AutotuneError::Io)?;

        if let Err(first_error) = fs::rename(&temporary, path) {
            if path.exists() {
                fs::remove_file(path).map_err(AutotuneError::Io)?;
                fs::rename(&temporary, path).map_err(AutotuneError::Io)?;
            } else {
                let _ = fs::remove_file(&temporary);
                return Err(AutotuneError::Io(first_error));
            }
        }
        Ok(())
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|value| value.to_os_string())
        .unwrap_or_else(|| "vrb-autotune.json".into());
    name.push(format!(".{}.tmp", std::process::id()));
    path.with_file_name(name)
}

#[derive(Debug, Error)]
pub enum AutotuneError {
    #[error("invalid empty environment fingerprint field '{0}'")]
    InvalidFingerprint(&'static str),
    #[error("workload shape must contain non-zero dimensions")]
    InvalidShape,
    #[error("workload shape element-count arithmetic overflow")]
    ShapeOverflow,
    #[error("workload alignment must be a non-zero power of two, got {0}")]
    InvalidAlignment(u32),
    #[error("benchmark latency must be finite and non-negative, got {0}")]
    InvalidLatency(f64),
    #[error("benchmark sample count must be non-zero")]
    ZeroSamples,
    #[error("unsupported autotune schema version {0}")]
    UnsupportedSchema(u32),
    #[error("autotune I/O error: {0}")]
    Io(#[source] std::io::Error),
    #[error("autotune JSON error: {0}")]
    Json(#[source] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use vrb_core::{BackendId, DataType, OperationKind};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn environment(runtime: &str) -> EnvironmentFingerprint {
        EnvironmentFingerprint {
            gpu_arch: "gfx1030".to_owned(),
            driver_version: "test-driver".to_owned(),
            runtime_version: runtime.to_owned(),
            plugin_version: "plugin-v1".to_owned(),
        }
    }

    fn shape() -> WorkloadShape {
        WorkloadShape {
            dims: vec![128, 128, 128],
            alignment: 16,
        }
    }

    fn record(runtime: &str, latency: f64) -> TuningRecord {
        TuningRecord {
            key: TuningKey {
                environment: environment(runtime),
                backend: BackendId::new("hip").unwrap(),
                operation: OperationKind::Gemm,
                data_type: DataType::F32,
                shape: shape(),
            },
            median_microseconds: latency,
            samples: 25,
        }
    }

    #[test]
    fn exact_environment_and_shape_project_into_core_table() {
        let mut database = AutotuneDatabase::default();
        database.record(record("runtime-a", 12.5)).unwrap();
        database.record(record("runtime-b", 99.0)).unwrap();

        let table = database
            .performance_table_for(&environment("runtime-a"), &shape())
            .unwrap();
        assert_eq!(
            table.median_us(
                &BackendId::new("hip").unwrap(),
                OperationKind::Gemm,
                DataType::F32
            ),
            Some(12.5)
        );
    }

    #[test]
    fn json_round_trip_preserves_records() {
        let mut database = AutotuneDatabase::default();
        database.record(record("runtime-a", 12.5)).unwrap();

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vrb-autotune-test-{}-{nonce}.json",
            std::process::id()
        ));
        database.save(&path).unwrap();
        let loaded = AutotuneDatabase::load(&path).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(loaded, database);
    }

    #[test]
    fn invalid_alignment_fails_closed() {
        let invalid = WorkloadShape {
            dims: vec![1, 2, 3],
            alignment: 3,
        };
        assert!(matches!(
            invalid.validate(),
            Err(AutotuneError::InvalidAlignment(3))
        ));
    }
}
