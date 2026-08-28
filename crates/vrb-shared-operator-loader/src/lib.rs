#![forbid(unsafe_op_in_unsafe_fn)]

use libloading::Library;
use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use vrb_core::BackendKind;
use vrb_operators::OperatorKind;
use vrb_shared_operator_plugin_api::{
    backend_kind, capability, expected_operator_info_struct_size, expected_plugin_struct_size,
    memory_handle_kind, operator_kind, resource_access, status, sync_handle_kind,
    ExecuteSharedOperatorFn, QuerySharedOperatorFn, SharedOperatorPluginEntryV1,
    ShutdownSharedOperatorPluginFn, VrbSharedOperatorExecutionRequestV1, VrbSharedOperatorInfoV1,
    VrbSharedOperatorPluginV1, VrbSharedResourceRegionV1, VrbSharedSyncPointV1,
    VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION, VRB_SHARED_OPERATOR_PLUGIN_ENTRY_SYMBOL,
};
use vrb_shared_operators::{
    ExternalMemoryHandleKind, ExternalSyncHandleKind, ResourceAccess, SharedOperator,
    SharedOperatorCapabilities, SharedOperatorError, SharedOperatorInvocation, SharedOperatorOutput,
    SharedOperatorRegistry, SharedResourceRegion, SharedSyncPoint,
};

const DEFAULT_MAX_OPERATORS: u32 = 4096;
const DEFAULT_MAX_METADATA_BYTES: u64 = 1024 * 1024;
const DEFAULT_MAX_RESOURCES: u32 = 1024;
const DEFAULT_MAX_SYNC_POINTS: u32 = 4096;
const DEFAULT_MAX_RECEIPT_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedOperatorLoadPolicy {
    pub max_operators: u32,
    pub max_metadata_bytes: u64,
    pub max_resources: u32,
    pub max_sync_points: u32,
    pub max_receipt_bytes: u64,
}

impl Default for SharedOperatorLoadPolicy {
    fn default() -> Self {
        Self {
            max_operators: DEFAULT_MAX_OPERATORS,
            max_metadata_bytes: DEFAULT_MAX_METADATA_BYTES,
            max_resources: DEFAULT_MAX_RESOURCES,
            max_sync_points: DEFAULT_MAX_SYNC_POINTS,
            max_receipt_bytes: DEFAULT_MAX_RECEIPT_BYTES,
        }
    }
}

#[derive(Debug, Error)]
pub enum SharedOperatorLoadError {
    #[error("unable to load shared-operator plugin '{path}': {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("shared-operator plugin entry symbol is missing: {0}")]
    MissingEntry(#[source] libloading::Error),
    #[error("shared-operator plugin entry returned a null descriptor")]
    NullDescriptor,
    #[error("shared-operator ABI version {actual} is incompatible with host ABI {expected}")]
    AbiVersion { actual: u32, expected: u32 },
    #[error("shared-operator descriptor is too small: {actual} bytes, expected at least {expected}")]
    DescriptorTooSmall { actual: u32, expected: u32 },
    #[error("shared-operator plugin name is empty")]
    EmptyPluginName,
    #[error("shared-operator plugin declares no operators")]
    NoOperators,
    #[error("shared-operator plugin declares {actual} operators, exceeding host limit {maximum}")]
    TooManyOperators { actual: u32, maximum: u32 },
    #[error("shared-operator plugin query callback is missing")]
    MissingQueryOperator,
    #[error("shared-operator plugin execute callback is missing")]
    MissingExecute,
    #[error("shared-operator query {index} returned status {status}")]
    QueryStatus { index: u32, status: i32 },
    #[error("shared-operator descriptor {index} is too small: {actual} bytes, expected at least {expected}")]
    OperatorDescriptorTooSmall {
        index: u32,
        actual: u32,
        expected: u32,
    },
    #[error("shared operator {index} has an empty name")]
    EmptyOperatorName { index: u32 },
    #[error("shared operator id {0} is duplicated within the plugin")]
    DuplicateOperatorId(u32),
    #[error("shared operator {operator_id} advertises invalid capabilities: {detail}")]
    InvalidCapabilities { operator_id: u32, detail: String },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SharedOperatorExecutionError {
    #[error("shared-operator plugin state lock is poisoned")]
    Poisoned,
    #[error("shared invocation metadata is {actual} bytes, exceeding host limit {maximum}")]
    MetadataTooLarge { actual: u64, maximum: u64 },
    #[error("shared invocation has {actual} resources, exceeding host limit {maximum}")]
    TooManyResources { actual: u32, maximum: u32 },
    #[error("shared invocation has {actual} synchronization points, exceeding host limit {maximum}")]
    TooManySyncPoints { actual: u32, maximum: u32 },
    #[error("shared invocation length cannot be represented by the ABI")]
    LengthUnsupported,
    #[error("shared operator plugin returned status {0}")]
    PluginStatus(i32),
    #[error("shared operator receipt length {actual} exceeds host capacity {capacity}")]
    ReceiptTooLarge { actual: u64, capacity: u64 },
    #[error("shared operator receipt capacity {0} cannot be represented on this host")]
    ReceiptCapacityUnsupported(u64),
    #[error(transparent)]
    InvalidInvocation(#[from] SharedOperatorError),
}

struct PluginCallbacks {
    user_data: usize,
    execute: ExecuteSharedOperatorFn,
    shutdown: Option<ShutdownSharedOperatorPluginFn>,
    shutdown_called: bool,
}

struct LoadedSharedPluginState {
    plugin_name: String,
    policy: SharedOperatorLoadPolicy,
    callbacks: Mutex<PluginCallbacks>,
    _library: Library,
}

impl Drop for LoadedSharedPluginState {
    fn drop(&mut self) {
        if let Ok(mut callbacks) = self.callbacks.lock() {
            if callbacks.shutdown_called {
                return;
            }
            if let Some(shutdown) = callbacks.shutdown {
                // SAFETY: shutdown came from a validated descriptor, is called at
                // most once, and the library is retained until this Drop completes.
                unsafe { shutdown(callbacks.user_data as *mut _) };
            }
            callbacks.shutdown_called = true;
        }
    }
}

pub struct DynamicSharedOperator {
    plugin: Arc<LoadedSharedPluginState>,
    operator_id: u32,
    name: String,
    capabilities: SharedOperatorCapabilities,
    raw_capability_bits: u64,
}

impl Debug for DynamicSharedOperator {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicSharedOperator")
            .field("plugin", &self.plugin.plugin_name)
            .field("operator_id", &self.operator_id)
            .field("name", &self.name)
            .field("capabilities", &self.capabilities)
            .field("raw_capability_bits", &self.raw_capability_bits)
            .finish()
    }
}

impl DynamicSharedOperator {
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

    fn execute_plugin(
        &self,
        invocation: SharedOperatorInvocation<'_>,
    ) -> Result<Vec<u8>, SharedOperatorExecutionError> {
        invocation.validate()?;
        let policy = self.plugin.policy;
        let metadata_len = u64::try_from(invocation.metadata.len())
            .map_err(|_| SharedOperatorExecutionError::LengthUnsupported)?;
        if metadata_len > policy.max_metadata_bytes {
            return Err(SharedOperatorExecutionError::MetadataTooLarge {
                actual: metadata_len,
                maximum: policy.max_metadata_bytes,
            });
        }

        let resource_count = u32::try_from(invocation.resources.len())
            .map_err(|_| SharedOperatorExecutionError::LengthUnsupported)?;
        if resource_count > policy.max_resources {
            return Err(SharedOperatorExecutionError::TooManyResources {
                actual: resource_count,
                maximum: policy.max_resources,
            });
        }
        let sync_count = invocation
            .waits
            .len()
            .checked_add(invocation.signals.len())
            .ok_or(SharedOperatorExecutionError::LengthUnsupported)?;
        let sync_count = u32::try_from(sync_count)
            .map_err(|_| SharedOperatorExecutionError::LengthUnsupported)?;
        if sync_count > policy.max_sync_points {
            return Err(SharedOperatorExecutionError::TooManySyncPoints {
                actual: sync_count,
                maximum: policy.max_sync_points,
            });
        }

        let abi_resources: Vec<VrbSharedResourceRegionV1> = invocation
            .resources
            .iter()
            .copied()
            .map(map_resource_to_abi)
            .collect();
        let abi_waits: Vec<VrbSharedSyncPointV1> = invocation
            .waits
            .iter()
            .copied()
            .map(map_sync_to_abi)
            .collect();
        let abi_signals: Vec<VrbSharedSyncPointV1> = invocation
            .signals
            .iter()
            .copied()
            .map(map_sync_to_abi)
            .collect();

        let receipt_capacity = usize::try_from(policy.max_receipt_bytes).map_err(|_| {
            SharedOperatorExecutionError::ReceiptCapacityUnsupported(policy.max_receipt_bytes)
        })?;
        let mut receipt = vec![0_u8; receipt_capacity];
        let mut receipt_len = 0_u64;

        let request = VrbSharedOperatorExecutionRequestV1 {
            operator_id: self.operator_id,
            metadata_ptr: slice_ptr(invocation.metadata),
            metadata_len,
            resources_ptr: slice_ptr(&abi_resources),
            resource_count,
            waits_ptr: slice_ptr(&abi_waits),
            wait_count: u32::try_from(abi_waits.len())
                .map_err(|_| SharedOperatorExecutionError::LengthUnsupported)?,
            signals_ptr: slice_ptr(&abi_signals),
            signal_count: u32::try_from(abi_signals.len())
                .map_err(|_| SharedOperatorExecutionError::LengthUnsupported)?,
            receipt_ptr: if receipt.is_empty() {
                std::ptr::null_mut()
            } else {
                receipt.as_mut_ptr()
            },
            receipt_capacity: policy.max_receipt_bytes,
            receipt_len: &mut receipt_len,
            ..VrbSharedOperatorExecutionRequestV1::default()
        };

        let callbacks = self
            .plugin
            .callbacks
            .lock()
            .map_err(|_| SharedOperatorExecutionError::Poisoned)?;
        // SAFETY: execute came from a validated plugin descriptor. All request
        // pointers refer to host-owned buffers that live for the complete call,
        // and callbacks sharing plugin state are serialized by the mutex.
        let execute_status =
            unsafe { (callbacks.execute)(callbacks.user_data as *mut _, &request as *const _) };
        if execute_status != status::OK {
            if execute_status == status::BUFFER_TOO_SMALL
                && receipt_len > policy.max_receipt_bytes
            {
                return Err(SharedOperatorExecutionError::ReceiptTooLarge {
                    actual: receipt_len,
                    capacity: policy.max_receipt_bytes,
                });
            }
            return Err(SharedOperatorExecutionError::PluginStatus(execute_status));
        }
        if receipt_len > policy.max_receipt_bytes {
            return Err(SharedOperatorExecutionError::ReceiptTooLarge {
                actual: receipt_len,
                capacity: policy.max_receipt_bytes,
            });
        }
        let actual = usize::try_from(receipt_len)
            .map_err(|_| SharedOperatorExecutionError::LengthUnsupported)?;
        receipt.truncate(actual);
        Ok(receipt)
    }
}

impl SharedOperator for DynamicSharedOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> SharedOperatorCapabilities {
        self.capabilities.clone()
    }

    fn execute_shared(
        &self,
        invocation: SharedOperatorInvocation<'_>,
    ) -> Result<SharedOperatorOutput, SharedOperatorError> {
        self.execute_plugin(invocation)
            .map(|receipt| SharedOperatorOutput { receipt })
            .map_err(|error| SharedOperatorError::Execution(error.to_string()))
    }
}

pub struct LoadedSharedOperatorLibrary {
    plugin_name: String,
    operators: Vec<Arc<DynamicSharedOperator>>,
}

impl Debug for LoadedSharedOperatorLibrary {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoadedSharedOperatorLibrary")
            .field("plugin_name", &self.plugin_name)
            .field("operators", &self.operators)
            .finish()
    }
}

impl LoadedSharedOperatorLibrary {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SharedOperatorLoadError> {
        Self::load_with_policy(path, SharedOperatorLoadPolicy::default())
    }

    pub fn load_with_policy(
        path: impl AsRef<Path>,
        policy: SharedOperatorLoadPolicy,
    ) -> Result<Self, SharedOperatorLoadError> {
        let path = path.as_ref().to_path_buf();
        // SAFETY: dynamic loading invokes the platform loader. Only the fixed
        // entry symbol is called before the descriptor is validated.
        let library = unsafe { Library::new(&path) }.map_err(|source| {
            SharedOperatorLoadError::Load {
                path: path.clone(),
                source,
            }
        })?;

        // SAFETY: symbol type is the documented fixed C ABI entry point.
        let descriptor = unsafe {
            let entry = library
                .get::<SharedOperatorPluginEntryV1>(VRB_SHARED_OPERATOR_PLUGIN_ENTRY_SYMBOL)
                .map_err(SharedOperatorLoadError::MissingEntry)?;
            let pointer = entry();
            if pointer.is_null() {
                return Err(SharedOperatorLoadError::NullDescriptor);
            }
            &*pointer
        };

        validate_plugin_descriptor(descriptor, policy)?;
        let plugin_name = abi_chars_to_string(&descriptor.name);
        let query_operator: QuerySharedOperatorFn = descriptor
            .query_operator
            .ok_or(SharedOperatorLoadError::MissingQueryOperator)?;
        let execute = descriptor
            .execute
            .ok_or(SharedOperatorLoadError::MissingExecute)?;
        let user_data = descriptor.user_data as usize;
        let plugin_capability_bits = descriptor.capability_bits;

        let mut discovered = Vec::with_capacity(descriptor.operator_count as usize);
        let mut ids = BTreeSet::new();
        for index in 0..descriptor.operator_count {
            let mut info = VrbSharedOperatorInfoV1::default();
            // SAFETY: query callback came from the validated descriptor and the
            // host provides a writable, self-sized info structure.
            let query_status =
                unsafe { query_operator(user_data as *mut _, index, &mut info as *mut _) };
            if query_status != status::OK {
                return Err(SharedOperatorLoadError::QueryStatus {
                    index,
                    status: query_status,
                });
            }
            let expected = expected_operator_info_struct_size();
            if info.struct_size < expected {
                return Err(SharedOperatorLoadError::OperatorDescriptorTooSmall {
                    index,
                    actual: info.struct_size,
                    expected,
                });
            }
            let name = abi_chars_to_string(&info.name);
            if name.is_empty() {
                return Err(SharedOperatorLoadError::EmptyOperatorName { index });
            }
            if !ids.insert(info.operator_id) {
                return Err(SharedOperatorLoadError::DuplicateOperatorId(info.operator_id));
            }

            let raw_capability_bits = info.capability_bits | plugin_capability_bits;
            validate_capabilities(info.operator_id, raw_capability_bits, &info)?;
            discovered.push((
                info.operator_id,
                name,
                raw_capability_bits,
                SharedOperatorCapabilities {
                    kind: map_operator_kind(info.operator_kind),
                    backend: map_backend_kind(info.backend_kind),
                    memory_kinds: map_memory_kinds(info.memory_kind_bits),
                    supports_external_synchronization: raw_capability_bits
                        & capability::EXTERNAL_SYNCHRONIZATION
                        != 0,
                    proven_zero_copy: raw_capability_bits & capability::PROVEN_ZERO_COPY != 0,
                },
            ));
        }

        let state = Arc::new(LoadedSharedPluginState {
            plugin_name: plugin_name.clone(),
            policy,
            callbacks: Mutex::new(PluginCallbacks {
                user_data,
                execute,
                shutdown: descriptor.shutdown,
                shutdown_called: false,
            }),
            _library: library,
        });
        let operators = discovered
            .into_iter()
            .map(|(operator_id, name, raw_capability_bits, capabilities)| {
                Arc::new(DynamicSharedOperator {
                    plugin: Arc::clone(&state),
                    operator_id,
                    name,
                    capabilities,
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
    pub fn operators(&self) -> &[Arc<DynamicSharedOperator>] {
        &self.operators
    }

    pub fn register_into(&self, registry: &mut SharedOperatorRegistry) {
        for operator in &self.operators {
            let injected: Arc<dyn SharedOperator> = operator.clone();
            registry.register(injected);
        }
    }
}

fn validate_plugin_descriptor(
    descriptor: &VrbSharedOperatorPluginV1,
    policy: SharedOperatorLoadPolicy,
) -> Result<(), SharedOperatorLoadError> {
    if descriptor.abi_version != VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION {
        return Err(SharedOperatorLoadError::AbiVersion {
            actual: descriptor.abi_version,
            expected: VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION,
        });
    }
    let expected = expected_plugin_struct_size();
    if descriptor.struct_size < expected {
        return Err(SharedOperatorLoadError::DescriptorTooSmall {
            actual: descriptor.struct_size,
            expected,
        });
    }
    if abi_chars_to_string(&descriptor.name).is_empty() {
        return Err(SharedOperatorLoadError::EmptyPluginName);
    }
    if descriptor.operator_count == 0 {
        return Err(SharedOperatorLoadError::NoOperators);
    }
    if descriptor.operator_count > policy.max_operators {
        return Err(SharedOperatorLoadError::TooManyOperators {
            actual: descriptor.operator_count,
            maximum: policy.max_operators,
        });
    }
    Ok(())
}

fn validate_capabilities(
    operator_id: u32,
    bits: u64,
    info: &VrbSharedOperatorInfoV1,
) -> Result<(), SharedOperatorLoadError> {
    if bits & capability::PROVEN_ZERO_COPY != 0 && bits & capability::EXTERNAL_RESOURCE == 0 {
        return Err(SharedOperatorLoadError::InvalidCapabilities {
            operator_id,
            detail: "proven zero-copy requires external-resource capability".to_owned(),
        });
    }
    if bits & capability::EXTERNAL_RESOURCE != 0 && info.memory_kind_bits == 0 {
        return Err(SharedOperatorLoadError::InvalidCapabilities {
            operator_id,
            detail: "external-resource capability requires at least one memory-handle kind"
                .to_owned(),
        });
    }
    if bits & capability::EXTERNAL_SYNCHRONIZATION != 0 && info.sync_kind_bits == 0 {
        return Err(SharedOperatorLoadError::InvalidCapabilities {
            operator_id,
            detail: "external-synchronization capability requires at least one sync-handle kind"
                .to_owned(),
        });
    }
    Ok(())
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

fn map_memory_kinds(bits: u64) -> Vec<ExternalMemoryHandleKind> {
    let candidates = [
        (memory_handle_kind::WIN32_KMT, ExternalMemoryHandleKind::Win32Kmt),
        (memory_handle_kind::WIN32_NT, ExternalMemoryHandleKind::Win32Nt),
        (memory_handle_kind::OPAQUE_FD, ExternalMemoryHandleKind::OpaqueFd),
        (memory_handle_kind::DMA_BUF, ExternalMemoryHandleKind::DmaBuf),
    ];
    candidates
        .into_iter()
        .filter_map(|(raw, kind)| ((bits & bit_for(raw)) != 0).then_some(kind))
        .collect()
}

fn map_resource_to_abi(resource: SharedResourceRegion) -> VrbSharedResourceRegionV1 {
    VrbSharedResourceRegionV1 {
        handle_kind: memory_kind_to_raw(resource.handle_kind),
        access: match resource.access {
            ResourceAccess::ReadOnly => resource_access::READ_ONLY,
            ResourceAccess::WriteOnly => resource_access::WRITE_ONLY,
            ResourceAccess::ReadWrite => resource_access::READ_WRITE,
        },
        handle: resource.handle,
        allocation_size: resource.allocation_size,
        offset: resource.offset,
        length: resource.length,
        ..VrbSharedResourceRegionV1::default()
    }
}

fn map_sync_to_abi(sync: SharedSyncPoint) -> VrbSharedSyncPointV1 {
    VrbSharedSyncPointV1 {
        handle_kind: sync_kind_to_raw(sync.handle_kind),
        handle: sync.handle,
        value: sync.value,
        ..VrbSharedSyncPointV1::default()
    }
}

fn memory_kind_to_raw(kind: ExternalMemoryHandleKind) -> u32 {
    match kind {
        ExternalMemoryHandleKind::Win32Kmt => memory_handle_kind::WIN32_KMT,
        ExternalMemoryHandleKind::Win32Nt => memory_handle_kind::WIN32_NT,
        ExternalMemoryHandleKind::OpaqueFd => memory_handle_kind::OPAQUE_FD,
        ExternalMemoryHandleKind::DmaBuf => memory_handle_kind::DMA_BUF,
        ExternalMemoryHandleKind::Custom(value) => value,
    }
}

fn sync_kind_to_raw(kind: ExternalSyncHandleKind) -> u32 {
    match kind {
        ExternalSyncHandleKind::Win32Opaque => sync_handle_kind::WIN32_OPAQUE,
        ExternalSyncHandleKind::OpaqueFd => sync_handle_kind::OPAQUE_FD,
        ExternalSyncHandleKind::Timeline => sync_handle_kind::TIMELINE,
        ExternalSyncHandleKind::Custom(value) => value,
    }
}

const fn bit_for(value: u32) -> u64 {
    if value < 64 {
        1_u64 << value
    } else {
        0
    }
}

fn slice_ptr<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        std::ptr::null()
    } else {
        slice.as_ptr()
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
    fn policy_is_bounded() {
        let policy = SharedOperatorLoadPolicy::default();
        assert!(policy.max_operators > 0);
        assert!(policy.max_metadata_bytes > 0);
        assert!(policy.max_resources > 0);
        assert!(policy.max_sync_points > 0);
        assert!(policy.max_receipt_bytes > 0);
    }

    #[test]
    fn unknown_operator_and_backend_tags_degrade_safely() {
        assert_eq!(map_operator_kind(u32::MAX), OperatorKind::Custom);
        assert_eq!(map_backend_kind(u32::MAX), BackendKind::Plugin);
    }

    #[test]
    fn proven_zero_copy_cannot_exist_without_external_resource_support() {
        let mut info = VrbSharedOperatorInfoV1::default();
        info.operator_id = 7;
        let error = validate_capabilities(7, capability::PROVEN_ZERO_COPY, &info)
            .expect_err("invalid capability combination must be rejected");
        assert!(matches!(
            error,
            SharedOperatorLoadError::InvalidCapabilities { operator_id: 7, .. }
        ));
    }
}
