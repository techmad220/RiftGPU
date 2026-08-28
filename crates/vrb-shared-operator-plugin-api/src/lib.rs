#![forbid(unsafe_op_in_unsafe_fn)]

//! Stable C ABI for dynamic operators that consume borrowed external resources.
//!
//! This ABI is independent from the host-byte operator ABI. Native handles are
//! borrowed for the duration of a callback only; plugins must not close, retain,
//! duplicate, or otherwise assume ownership of them unless a future ABI version
//! explicitly adds such a transfer.

use std::ffi::{c_char, c_void};

pub const VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION: u32 = 1;
pub const VRB_SHARED_OPERATOR_PLUGIN_ENTRY_SYMBOL: &[u8] =
    b"vrb_shared_operator_plugin_entry_v1\0";
pub const VRB_SHARED_OPERATOR_NAME_CAPACITY: usize = 128;
pub const VRB_SHARED_OPERATOR_PLUGIN_NAME_CAPACITY: usize = 128;

pub mod status {
    pub const OK: i32 = 0;
    pub const INVALID_ARGUMENT: i32 = 1;
    pub const UNSUPPORTED: i32 = 2;
    pub const UNAVAILABLE: i32 = 3;
    pub const BUFFER_TOO_SMALL: i32 = 4;
    pub const INTERNAL_ERROR: i32 = 5;
}

pub mod operator_kind {
    pub const GEMM: u32 = 1;
    pub const ATTENTION: u32 = 2;
    pub const QUANTIZE: u32 = 3;
    pub const DEQUANTIZE: u32 = 4;
    pub const TRANSFORM: u32 = 5;
    pub const CUSTOM: u32 = 255;
}

pub mod backend_kind {
    pub const CPU: u32 = 1;
    pub const VULKAN: u32 = 2;
    pub const HIP: u32 = 3;
    pub const HYBRID: u32 = 4;
    pub const PLUGIN: u32 = 255;
}

pub mod capability {
    pub const EXTERNAL_RESOURCE: u64 = 1 << 0;
    pub const EXTERNAL_SYNCHRONIZATION: u64 = 1 << 1;
    /// Set only after the implementation's actual execution path has been
    /// certified to avoid host-relay copies for bulk tensor payloads.
    pub const PROVEN_ZERO_COPY: u64 = 1 << 2;
    pub const FP16: u64 = 1 << 3;
    pub const FP32: u64 = 1 << 4;
    pub const INT8: u64 = 1 << 5;
}

pub mod memory_handle_kind {
    pub const WIN32_KMT: u32 = 1;
    pub const WIN32_NT: u32 = 2;
    pub const OPAQUE_FD: u32 = 3;
    pub const DMA_BUF: u32 = 4;
    pub const CUSTOM_BASE: u32 = 0x8000_0000;
}

pub mod resource_access {
    pub const READ_ONLY: u32 = 1;
    pub const WRITE_ONLY: u32 = 2;
    pub const READ_WRITE: u32 = 3;
}

pub mod sync_handle_kind {
    pub const WIN32_OPAQUE: u32 = 1;
    pub const OPAQUE_FD: u32 = 2;
    pub const TIMELINE: u32 = 3;
    pub const CUSTOM_BASE: u32 = 0x8000_0000;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VrbSharedResourceRegionV1 {
    pub struct_size: u32,
    pub handle_kind: u32,
    pub access: u32,
    pub reserved0: u32,
    pub handle: u64,
    pub allocation_size: u64,
    pub offset: u64,
    pub length: u64,
    pub reserved: [u64; 4],
}

impl Default for VrbSharedResourceRegionV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            handle_kind: 0,
            access: 0,
            reserved0: 0,
            handle: 0,
            allocation_size: 0,
            offset: 0,
            length: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VrbSharedSyncPointV1 {
    pub struct_size: u32,
    pub handle_kind: u32,
    pub handle: u64,
    pub value: u64,
    pub flags: u64,
    pub reserved: [u64; 4],
}

impl Default for VrbSharedSyncPointV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            handle_kind: 0,
            handle: 0,
            value: 0,
            flags: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VrbSharedOperatorInfoV1 {
    pub struct_size: u32,
    pub operator_id: u32,
    pub operator_kind: u32,
    pub backend_kind: u32,
    pub capability_bits: u64,
    /// Bit N advertises support for memory_handle_kind value N when N < 64.
    pub memory_kind_bits: u64,
    /// Bit N advertises support for sync_handle_kind value N when N < 64.
    pub sync_kind_bits: u64,
    pub name: [c_char; VRB_SHARED_OPERATOR_NAME_CAPACITY],
    pub reserved: [u64; 4],
}

impl Default for VrbSharedOperatorInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            operator_id: 0,
            operator_kind: operator_kind::CUSTOM,
            backend_kind: backend_kind::PLUGIN,
            capability_bits: 0,
            memory_kind_bits: 0,
            sync_kind_bits: 0,
            name: [0; VRB_SHARED_OPERATOR_NAME_CAPACITY],
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VrbSharedOperatorExecutionRequestV1 {
    pub struct_size: u32,
    pub operator_id: u32,
    pub flags: u64,
    pub metadata_ptr: *const u8,
    pub metadata_len: u64,
    pub resources_ptr: *const VrbSharedResourceRegionV1,
    pub resource_count: u32,
    pub reserved0: u32,
    pub waits_ptr: *const VrbSharedSyncPointV1,
    pub wait_count: u32,
    pub reserved1: u32,
    pub signals_ptr: *const VrbSharedSyncPointV1,
    pub signal_count: u32,
    pub reserved2: u32,
    pub receipt_ptr: *mut u8,
    pub receipt_capacity: u64,
    pub receipt_len: *mut u64,
    pub reserved: [u64; 4],
}

impl Default for VrbSharedOperatorExecutionRequestV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            operator_id: 0,
            flags: 0,
            metadata_ptr: std::ptr::null(),
            metadata_len: 0,
            resources_ptr: std::ptr::null(),
            resource_count: 0,
            reserved0: 0,
            waits_ptr: std::ptr::null(),
            wait_count: 0,
            reserved1: 0,
            signals_ptr: std::ptr::null(),
            signal_count: 0,
            reserved2: 0,
            receipt_ptr: std::ptr::null_mut(),
            receipt_capacity: 0,
            receipt_len: std::ptr::null_mut(),
            reserved: [0; 4],
        }
    }
}

pub type QuerySharedOperatorFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    index: u32,
    out_info: *mut VrbSharedOperatorInfoV1,
) -> i32;

pub type ExecuteSharedOperatorFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    request: *const VrbSharedOperatorExecutionRequestV1,
) -> i32;

pub type ShutdownSharedOperatorPluginFn = unsafe extern "C" fn(user_data: *mut c_void);

#[repr(C)]
pub struct VrbSharedOperatorPluginV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub name: [c_char; VRB_SHARED_OPERATOR_PLUGIN_NAME_CAPACITY],
    pub operator_count: u32,
    pub reserved0: u32,
    pub capability_bits: u64,
    pub user_data: *mut c_void,
    pub query_operator: Option<QuerySharedOperatorFn>,
    pub execute: Option<ExecuteSharedOperatorFn>,
    pub shutdown: Option<ShutdownSharedOperatorPluginFn>,
    pub reserved: [usize; 5],
}

pub type SharedOperatorPluginEntryV1 = unsafe extern "C" fn() -> *const VrbSharedOperatorPluginV1;

#[must_use]
pub const fn expected_resource_region_struct_size() -> u32 {
    std::mem::size_of::<VrbSharedResourceRegionV1>() as u32
}

#[must_use]
pub const fn expected_sync_point_struct_size() -> u32 {
    std::mem::size_of::<VrbSharedSyncPointV1>() as u32
}

#[must_use]
pub const fn expected_operator_info_struct_size() -> u32 {
    std::mem::size_of::<VrbSharedOperatorInfoV1>() as u32
}

#[must_use]
pub const fn expected_execution_request_struct_size() -> u32 {
    std::mem::size_of::<VrbSharedOperatorExecutionRequestV1>() as u32
}

#[must_use]
pub const fn expected_plugin_struct_size() -> u32 {
    std::mem::size_of::<VrbSharedOperatorPluginV1>() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_and_defaults_are_self_describing() {
        assert_eq!(VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION, 1);
        assert_eq!(
            VrbSharedResourceRegionV1::default().struct_size,
            expected_resource_region_struct_size()
        );
        assert_eq!(
            VrbSharedSyncPointV1::default().struct_size,
            expected_sync_point_struct_size()
        );
        assert_eq!(
            VrbSharedOperatorInfoV1::default().struct_size,
            expected_operator_info_struct_size()
        );
        assert_eq!(
            VrbSharedOperatorExecutionRequestV1::default().struct_size,
            expected_execution_request_struct_size()
        );
        assert!(expected_plugin_struct_size() > 0);
    }

    #[test]
    fn proven_zero_copy_is_a_distinct_capability() {
        assert_ne!(capability::EXTERNAL_RESOURCE, capability::PROVEN_ZERO_COPY);
        assert_eq!(
            capability::EXTERNAL_RESOURCE & capability::PROVEN_ZERO_COPY,
            0
        );
    }
}
