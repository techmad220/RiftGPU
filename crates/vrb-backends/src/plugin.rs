use libloading::Library;
use std::fmt::{Debug, Formatter};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use thiserror::Error;
use vrb_core::{
    BackendError, BackendId, BackendKind, BackendProbe, CapabilitySet, ComputeBackend, DataType,
    OperationKind,
};
use vrb_plugin_api::{
    capability, expected_plugin_struct_size, ExecuteFn, PluginEntryV1, ProbeFn, ShutdownFn,
    VrbBackendInfoV1, VrbBackendKind, VrbExecutionRequestV1, VrbStatus, VRB_PLUGIN_ABI_VERSION,
    VRB_PLUGIN_ENTRY_SYMBOL,
};

#[derive(Debug, Error)]
pub enum PluginLoadError {
    #[error("unable to load plugin '{path}': {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("plugin entry symbol is missing: {0}")]
    MissingEntry(#[source] libloading::Error),
    #[error("plugin entry returned a null descriptor")]
    NullDescriptor,
    #[error("plugin ABI version {actual} is incompatible with host ABI {expected}")]
    AbiVersion { actual: u32, expected: u32 },
    #[error("plugin descriptor is too small: {actual} bytes, expected at least {expected}")]
    DescriptorTooSmall { actual: u32, expected: u32 },
    #[error("plugin name pointer is null")]
    NullName,
    #[error("plugin name is empty")]
    EmptyName,
    #[error("plugin probe callback is missing")]
    MissingProbe,
    #[error("plugin execute callback is missing")]
    MissingExecute,
    #[error("plugin returned status {0:?}")]
    PluginStatus(VrbStatus),
    #[error("plugin state lock is poisoned")]
    Poisoned,
    #[error(transparent)]
    Runtime(#[from] vrb_core::RuntimeError),
}

struct PluginState {
    user_data: usize,
    probe: ProbeFn,
    execute: Option<ExecuteFn>,
    shutdown: Option<ShutdownFn>,
    shutdown_called: bool,
}

pub struct DynamicPluginBackend {
    id: BackendId,
    name: String,
    kind: BackendKind,
    capability_bits: u64,
    state: Mutex<PluginState>,
    // Keep the library alive until after Drop invokes the optional shutdown hook.
    _library: Library,
}

impl Debug for DynamicPluginBackend {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicPluginBackend")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("capability_bits", &self.capability_bits)
            .finish_non_exhaustive()
    }
}

impl DynamicPluginBackend {
    /// Load a backend plugin from a dynamic library.
    ///
    /// The plugin boundary is a versioned C ABI. Rust trait objects, `String`,
    /// `Vec`, and other unstable Rust-layout types never cross the DLL boundary.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PluginLoadError> {
        let path = path.as_ref().to_path_buf();
        // SAFETY: loading a plugin necessarily executes platform loader behavior.
        // The caller chooses the plugin path; ABI validation occurs before any
        // plugin callback other than the fixed entry point is accepted.
        let library = unsafe { Library::new(&path) }.map_err(|source| PluginLoadError::Load {
            path: path.clone(),
            source,
        })?;

        // SAFETY: the symbol has a fixed C ABI and is validated immediately.
        let descriptor = unsafe {
            let entry = library
                .get::<PluginEntryV1>(VRB_PLUGIN_ENTRY_SYMBOL)
                .map_err(PluginLoadError::MissingEntry)?;
            let pointer = entry();
            if pointer.is_null() {
                return Err(PluginLoadError::NullDescriptor);
            }
            &*pointer
        };

        if descriptor.abi_version != VRB_PLUGIN_ABI_VERSION {
            return Err(PluginLoadError::AbiVersion {
                actual: descriptor.abi_version,
                expected: VRB_PLUGIN_ABI_VERSION,
            });
        }
        let expected_size = expected_plugin_struct_size();
        if descriptor.struct_size < expected_size {
            return Err(PluginLoadError::DescriptorTooSmall {
                actual: descriptor.struct_size,
                expected: expected_size,
            });
        }
        if descriptor.name.is_null() {
            return Err(PluginLoadError::NullName);
        }

        // SAFETY: a conforming plugin must return a NUL-terminated static name.
        let name = unsafe { std::ffi::CStr::from_ptr(descriptor.name) }
            .to_string_lossy()
            .trim()
            .to_owned();
        if name.is_empty() {
            return Err(PluginLoadError::EmptyName);
        }

        let probe = descriptor.probe.ok_or(PluginLoadError::MissingProbe)?;
        let id = BackendId::new(format!("plugin:{name}"))?;
        Ok(Self {
            id,
            name,
            kind: map_backend_kind(descriptor.backend_kind),
            capability_bits: descriptor.capability_bits,
            state: Mutex::new(PluginState {
                user_data: descriptor.user_data as usize,
                probe,
                execute: descriptor.execute,
                shutdown: descriptor.shutdown,
                shutdown_called: false,
            }),
            _library: library,
        })
    }

    pub fn execute_raw(&self, request: &VrbExecutionRequestV1) -> Result<(), PluginLoadError> {
        let state = self.state.lock().map_err(|_| PluginLoadError::Poisoned)?;
        let execute = state.execute.ok_or(PluginLoadError::MissingExecute)?;
        // SAFETY: callback and user_data were supplied by a validated plugin
        // descriptor and calls are serialized through the state mutex.
        let status = unsafe { execute(state.user_data as *mut _, request) };
        status_result(status)
    }

    fn probe_plugin(&self) -> Result<VrbBackendInfoV1, PluginLoadError> {
        let state = self.state.lock().map_err(|_| PluginLoadError::Poisoned)?;
        let mut info = VrbBackendInfoV1::default();
        // SAFETY: callback and user_data came from the validated descriptor. The
        // writable output points to a correctly sized host-owned ABI structure.
        let status = unsafe { (state.probe)(state.user_data as *mut _, &mut info) };
        status_result(status)?;
        Ok(info)
    }
}

impl Drop for DynamicPluginBackend {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if !state.shutdown_called {
                if let Some(shutdown) = state.shutdown {
                    // SAFETY: shutdown is invoked once, while the library is still
                    // loaded, and with the exact user_data from the descriptor.
                    unsafe { shutdown(state.user_data as *mut _) };
                }
                state.shutdown_called = true;
            }
        }
    }
}

impl ComputeBackend for DynamicPluginBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn kind(&self) -> BackendKind {
        self.kind
    }

    fn probe(&self) -> Result<BackendProbe, BackendError> {
        let info = self
            .probe_plugin()
            .map_err(|error| BackendError::Probe(error.to_string()))?;
        let name = abi_chars_to_string(&info.name);
        let vendor = abi_chars_to_string(&info.vendor);
        let bits = info.capability_bits | self.capability_bits;
        let data_types = data_types_from_bits(bits);

        Ok(BackendProbe {
            id: self.id.clone(),
            kind: map_backend_kind(info.backend_kind),
            name: if name.is_empty() {
                self.name.clone()
            } else {
                name
            },
            vendor,
            available: info.device_count > 0,
            device_count: info.device_count,
            detail: format!("plugin ABI v{VRB_PLUGIN_ABI_VERSION}"),
            capabilities: CapabilitySet {
                operations: vec![OperationKind::Custom],
                data_types,
                external_memory: bits & capability::EXTERNAL_MEMORY != 0,
                external_semaphore: bits & capability::EXTERNAL_SEMAPHORE != 0,
                zero_copy: bits & capability::ZERO_COPY != 0,
            },
        })
    }
}

fn map_backend_kind(kind: VrbBackendKind) -> BackendKind {
    match kind {
        VrbBackendKind::Cpu => BackendKind::Cpu,
        VrbBackendKind::Vulkan => BackendKind::Vulkan,
        VrbBackendKind::Hip => BackendKind::Hip,
        VrbBackendKind::Hybrid => BackendKind::Hybrid,
        VrbBackendKind::Other => BackendKind::Plugin,
    }
}

fn data_types_from_bits(bits: u64) -> Vec<DataType> {
    let mut types = Vec::new();
    if bits & capability::FP32 != 0 {
        types.push(DataType::F32);
    }
    if bits & capability::FP16 != 0 {
        types.push(DataType::F16);
    }
    if bits & capability::INT8 != 0 {
        types.push(DataType::I8);
    }
    if types.is_empty() {
        types.push(DataType::Unknown);
    }
    types
}

fn abi_chars_to_string<const N: usize>(value: &[std::ffi::c_char; N]) -> String {
    let bytes: Vec<u8> = value
        .iter()
        .copied()
        .take_while(|character| *character != 0)
        .map(|character| character as u8)
        .collect();
    String::from_utf8_lossy(&bytes).trim().to_owned()
}

fn status_result(status: VrbStatus) -> Result<(), PluginLoadError> {
    match status {
        VrbStatus::Ok => Ok(()),
        other => Err(PluginLoadError::PluginStatus(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_bits_map_to_data_types() {
        assert_eq!(
            data_types_from_bits(capability::FP32 | capability::FP16),
            vec![DataType::F32, DataType::F16]
        );
        assert_eq!(data_types_from_bits(0), vec![DataType::Unknown]);
    }

    #[test]
    fn abi_char_buffer_conversion_stops_at_nul() {
        let mut value = [0 as std::ffi::c_char; 8];
        value[0] = b'A' as std::ffi::c_char;
        value[1] = b'M' as std::ffi::c_char;
        value[2] = b'D' as std::ffi::c_char;
        assert_eq!(abi_chars_to_string(&value), "AMD");
    }
}
