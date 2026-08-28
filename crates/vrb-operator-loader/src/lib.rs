#![forbid(unsafe_op_in_unsafe_fn)]

use libloading::Library;
use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use vrb_core::BackendKind;
use vrb_operator_plugin_api::{
    backend_kind, expected_operator_info_struct_size, expected_operator_plugin_struct_size,
    operator_kind, status, ExecuteOperatorFn, OperatorPluginEntryV1, OutputSizeFn,
    ShutdownOperatorPluginFn, VrbOperatorExecutionRequestV1, VrbOperatorInfoV1,
    VRB_OPERATOR_PLUGIN_ABI_VERSION, VRB_OPERATOR_PLUGIN_ENTRY_SYMBOL,
};
use vrb_operators::{
    Operator, OperatorCapabilities, OperatorError, OperatorInvocation, OperatorKind, OperatorOutput,
    OperatorRegistry,
};

const DEFAULT_MAX_OPERATORS: u32 = 4096;
const DEFAULT_MAX_OUTPUT_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatorPluginLoadPolicy {
    pub max_operators: u32,
    pub max_output_bytes: u64,
}

impl Default for OperatorPluginLoadPolicy {
    fn default() -> Self {
        Self {
            max_operators: DEFAULT_MAX_OPERATORS,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }
}

#[derive(Debug, Error)]
pub enum OperatorPluginLoadError {
    #[error("unable to load operator plugin '{path}': {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("operator plugin entry symbol is missing: {0}")]
    MissingEntry(#[source] libloading::Error),
    #[error("operator plugin entry returned a null descriptor")]
    NullDescriptor,
    #[error("operator plugin ABI version {actual} is incompatible with host ABI {expected}")]
    AbiVersion { actual: u32, expected: u32 },
    #[error("operator plugin descriptor is too small: {actual} bytes, expected at least {expected}")]
    DescriptorTooSmall { actual: u32, expected: u32 },
    #[error("operator plugin name is empty")]
    EmptyPluginName,
    #[error("operator plugin declares no operators")]
    NoOperators,
    #[error("operator plugin declares {actual} operators, exceeding host limit {maximum}")]
    TooManyOperators { actual: u32, maximum: u32 },
    #[error("operator plugin query callback is missing")]
    MissingQueryOperator,
    #[error("operator plugin output-size callback is missing")]
    MissingOutputSize,
    #[error("operator plugin execute callback is missing")]
    MissingExecute,
    #[error("operator query {index} returned status {status}")]
    QueryStatus { index: u32, status: i32 },
    #[error("operator descriptor {index} is too small: {actual} bytes, expected at least {expected}")]
    OperatorDescriptorTooSmall {
        index: u32,
        actual: u32,
        expected: u32,
    },
    #[error("operator {index} has an empty name")]
    EmptyOperatorName { index: u32 },
    #[error("operator id {0} is duplicated within the plugin")]
    DuplicateOperatorId(u32),
    #[error("operator plugin state lock is poisoned")]
    Poisoned,
}

struct PluginCallbacks {
    user_data: usize,
    output_size: OutputSizeFn,
    execute: ExecuteOperatorFn,
    shutdown: Option<ShutdownOperatorPluginFn>,
    shutdown_called: bool,
}

struct LoadedPluginState {
    plugin_name: String,
    max_output_bytes: u64,
    callbacks: Mutex<PluginCallbacks>,
    // The library must outlive every callback and the optional shutdown call.
    _library: Library,
}

impl Drop for LoadedPluginState {
    fn drop(&mut self) {
        if let Ok(mut callbacks) = self.callbacks.lock() {
            if callbacks.shutdown_called {
                return;
            }
            if let Some(shutdown) = callbacks.shutdown {
                // SAFETY: shutdown came from a validated descriptor, is called
                // at most once, and the library is still loaded during Drop.
                unsafe { shutdown(callbacks.user_data as *mut _) };
            }
            callbacks.shutdown_called = true;
        }
    }
}

pub struct DynamicOperator {
    plugin: Arc<LoadedPluginState>,
    operator_id: u32,
    name: String,
    kind: OperatorKind,
    backend: BackendKind,
    raw_capability_bits: u64,
}

impl Debug for DynamicOperator {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicOperator")
            .field("plugin", &self.plugin.plugin_name)
            .field("operator_id", &self.operator_id)
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("backend", &self.backend)
            .field("raw_capability_bits", &self.raw_capability_bits)
            .finish()
    }
}

impl DynamicOperator {
    #[must_use]
    pub const fn operator_id(&self) -> u32 {
        self.operator_id
    }

    #[must_use]
    pub const fn raw_capability_bits(&self) -> u64 {
        self.raw_capability_bits
    }

    #[must_use]
    pub fn plugin_name(&self) -> &str {
        &self.plugin.plugin_name
    }

    fn execute_plugin(&self, input: &[u8]) -> Result<Vec<u8>, OperatorPluginExecutionError> {
        let callbacks = self
            .plugin
            .callbacks
            .lock()
            .map_err(|_| OperatorPluginExecutionError::Poisoned)?;

        let input_len = u64::try_from(input.len())
            .map_err(|_| OperatorPluginExecutionError::InputTooLarge(input.len()))?;
        let mut output_len = 0_u64;

        // SAFETY: callback and user_data came from a validated plugin descriptor.
        // The input slice and output_len pointer remain valid for the call.
        let size_status = unsafe {
            (callbacks.output_size)(
                callbacks.user_data as *mut _,
                self.operator_id,
                input.as_ptr(),
                input_len,
                &mut output_len,
            )
        };
        if size_status != status::OK {
            return Err(OperatorPluginExecutionError::PluginStatus(size_status));
        }
        if output_len > self.plugin.max_output_bytes {
            return Err(OperatorPluginExecutionError::OutputTooLarge {
                requested: output_len,
                maximum: self.plugin.max_output_bytes,
            });
        }

        let output_capacity = usize::try_from(output_len)
            .map_err(|_| OperatorPluginExecutionError::OutputLengthUnsupported(output_len))?;
        let mut output = vec![0_u8; output_capacity];
        let mut actual_output_len = output_len;
        let output_ptr = if output.is_empty() {
            std::ptr::null_mut()
        } else {
            output.as_mut_ptr()
        };
        let request = VrbOperatorExecutionRequestV1 {
            operator_id: self.operator_id,
            input_ptr: input.as_ptr(),
            input_len,
            output_ptr,
            output_capacity: output_len,
            output_len: &mut actual_output_len,
            ..VrbOperatorExecutionRequestV1::default()
        };

        // SAFETY: callback and user_data came from a validated plugin descriptor.
        // The request and all buffers it references remain valid for the call,
        // and plugin callbacks are serialized through the mutex.
        let execute_status = unsafe {
            (callbacks.execute)(callbacks.user_data as *mut _, &request as *const _)
        };
        if execute_status != status::OK {
            return Err(OperatorPluginExecutionError::PluginStatus(execute_status));
        }
        if actual_output_len > output_len {
            return Err(OperatorPluginExecutionError::OutputLengthExceeded {
                actual: actual_output_len,
                capacity: output_len,
            });
        }

        let actual = usize::try_from(actual_output_len).map_err(|_| {
            OperatorPluginExecutionError::OutputLengthUnsupported(actual_output_len)
        })?;
        output.truncate(actual);
        Ok(output)
    }
}

impl Operator for DynamicOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> OperatorCapabilities {
        OperatorCapabilities {
            kind: self.kind,
            backend: self.backend,
            // ABI v1's host-byte adapter does not pass shared-resource handles.
            // Never advertise zero-copy until an invocation path truly preserves it.
            supports_zero_copy: false,
        }
    }

    fn execute(&self, invocation: OperatorInvocation<'_>) -> Result<OperatorOutput, OperatorError> {
        self.execute_plugin(invocation.input)
            .map(|bytes| OperatorOutput { bytes })
            .map_err(|error| OperatorError::Execution(error.to_string()))
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum OperatorPluginExecutionError {
    #[error("operator plugin state lock is poisoned")]
    Poisoned,
    #[error("input length {0} cannot be represented by the plugin ABI")]
    InputTooLarge(usize),
    #[error("plugin returned status {0}")]
    PluginStatus(i32),
    #[error("plugin requested {requested} output bytes, exceeding host limit {maximum}")]
    OutputTooLarge { requested: u64, maximum: u64 },
    #[error("output length {0} cannot be represented on this host")]
    OutputLengthUnsupported(u64),
    #[error("plugin reported {actual} output bytes after receiving capacity {capacity}")]
    OutputLengthExceeded { actual: u64, capacity: u64 },
}

pub struct LoadedOperatorLibrary {
    plugin_name: String,
    operators: Vec<Arc<DynamicOperator>>,
}

impl Debug for LoadedOperatorLibrary {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoadedOperatorLibrary")
            .field("plugin_name", &self.plugin_name)
            .field("operators", &self.operators)
            .finish()
    }
}

impl LoadedOperatorLibrary {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, OperatorPluginLoadError> {
        Self::load_with_policy(path, OperatorPluginLoadPolicy::default())
    }

    pub fn load_with_policy(
        path: impl AsRef<Path>,
        policy: OperatorPluginLoadPolicy,
    ) -> Result<Self, OperatorPluginLoadError> {
        let path = path.as_ref().to_path_buf();
        // SAFETY: dynamic loading inherently executes platform loader behavior.
        // The fixed entry point is the only symbol called before ABI validation.
        let library = unsafe { Library::new(&path) }.map_err(|source| {
            OperatorPluginLoadError::Load {
                path: path.clone(),
                source,
            }
        })?;

        // SAFETY: the symbol type is the documented fixed C ABI entry point.
        let descriptor = unsafe {
            let entry = library
                .get::<OperatorPluginEntryV1>(VRB_OPERATOR_PLUGIN_ENTRY_SYMBOL)
                .map_err(OperatorPluginLoadError::MissingEntry)?;
            let pointer = entry();
            if pointer.is_null() {
                return Err(OperatorPluginLoadError::NullDescriptor);
            }
            &*pointer
        };

        if descriptor.abi_version != VRB_OPERATOR_PLUGIN_ABI_VERSION {
            return Err(OperatorPluginLoadError::AbiVersion {
                actual: descriptor.abi_version,
                expected: VRB_OPERATOR_PLUGIN_ABI_VERSION,
            });
        }
        let expected_plugin_size = expected_operator_plugin_struct_size();
        if descriptor.struct_size < expected_plugin_size {
            return Err(OperatorPluginLoadError::DescriptorTooSmall {
                actual: descriptor.struct_size,
                expected: expected_plugin_size,
            });
        }

        let plugin_name = abi_chars_to_string(&descriptor.name);
        if plugin_name.is_empty() {
            return Err(OperatorPluginLoadError::EmptyPluginName);
        }
        if descriptor.operator_count == 0 {
            return Err(OperatorPluginLoadError::NoOperators);
        }
        if descriptor.operator_count > policy.max_operators {
            return Err(OperatorPluginLoadError::TooManyOperators {
                actual: descriptor.operator_count,
                maximum: policy.max_operators,
            });
        }

        let query_operator = descriptor
            .query_operator
            .ok_or(OperatorPluginLoadError::MissingQueryOperator)?;
        let output_size = descriptor
            .output_size
            .ok_or(OperatorPluginLoadError::MissingOutputSize)?;
        let execute = descriptor
            .execute
            .ok_or(OperatorPluginLoadError::MissingExecute)?;
        let user_data = descriptor.user_data as usize;
        let plugin_capability_bits = descriptor.capability_bits;

        let mut discovered = Vec::with_capacity(descriptor.operator_count as usize);
        let mut ids = BTreeSet::new();
        for index in 0..descriptor.operator_count {
            let mut info = VrbOperatorInfoV1::default();
            // SAFETY: query_operator came from the validated descriptor. The host
            // provides a writable, correctly sized VrbOperatorInfoV1.
            let query_status = unsafe {
                query_operator(user_data as *mut _, index, &mut info as *mut _)
            };
            if query_status != status::OK {
                return Err(OperatorPluginLoadError::QueryStatus {
                    index,
                    status: query_status,
                });
            }
            let expected_info_size = expected_operator_info_struct_size();
            if info.struct_size < expected_info_size {
                return Err(OperatorPluginLoadError::OperatorDescriptorTooSmall {
                    index,
                    actual: info.struct_size,
                    expected: expected_info_size,
                });
            }
            let name = abi_chars_to_string(&info.name);
            if name.is_empty() {
                return Err(OperatorPluginLoadError::EmptyOperatorName { index });
            }
            if !ids.insert(info.operator_id) {
                return Err(OperatorPluginLoadError::DuplicateOperatorId(info.operator_id));
            }

            discovered.push((
                info.operator_id,
                name,
                map_operator_kind(info.operator_kind),
                map_backend_kind(info.backend_kind),
                info.capability_bits | plugin_capability_bits,
            ));
        }

        let state = Arc::new(LoadedPluginState {
            plugin_name: plugin_name.clone(),
            max_output_bytes: policy.max_output_bytes,
            callbacks: Mutex::new(PluginCallbacks {
                user_data,
                output_size,
                execute,
                shutdown: descriptor.shutdown,
                shutdown_called: false,
            }),
            _library: library,
        });

        let operators = discovered
            .into_iter()
            .map(|(operator_id, name, kind, backend, raw_capability_bits)| {
                Arc::new(DynamicOperator {
                    plugin: Arc::clone(&state),
                    operator_id,
                    name,
                    kind,
                    backend,
                    raw_capability_bits,
                })
            })
            .collect();

        Ok(Self {
            plugin_name,
            operators,
        })
    }

    #[must_use]
    pub fn plugin_name(&self) -> &str {
        &self.plugin_name
    }

    #[must_use]
    pub fn operators(&self) -> &[Arc<DynamicOperator>] {
        &self.operators
    }

    pub fn register_into(&self, registry: &mut OperatorRegistry) {
        for operator in &self.operators {
            let injected: Arc<dyn Operator> = Arc::clone(operator) as Arc<dyn Operator>;
            registry.register(injected);
        }
    }
}

fn map_operator_kind(kind: u32) -> OperatorKind {
    match kind {
        operator_kind::GEMM => OperatorKind::Gemm,
        operator_kind::ATTENTION => OperatorKind::Attention,
        operator_kind::QUANTIZE => OperatorKind::Quantize,
        operator_kind::DEQUANTIZE => OperatorKind::Dequantize,
        operator_kind::TRANSFORM => OperatorKind::Transform,
        _ => OperatorKind::Custom,
    }
}

fn map_backend_kind(kind: u32) -> BackendKind {
    match kind {
        backend_kind::CPU => BackendKind::Cpu,
        backend_kind::VULKAN => BackendKind::Vulkan,
        backend_kind::HIP => BackendKind::Hip,
        backend_kind::HYBRID => BackendKind::Hybrid,
        _ => BackendKind::Plugin,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_tags_degrade_without_invalid_ffi_enums() {
        assert_eq!(map_operator_kind(u32::MAX), OperatorKind::Custom);
        assert_eq!(map_backend_kind(u32::MAX), BackendKind::Plugin);
    }

    #[test]
    fn load_policy_has_bounded_defaults() {
        let policy = OperatorPluginLoadPolicy::default();
        assert!(policy.max_operators > 0);
        assert!(policy.max_output_bytes > 0);
    }
}
