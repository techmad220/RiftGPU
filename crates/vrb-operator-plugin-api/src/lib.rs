#![forbid(unsafe_op_in_unsafe_fn)]

//! Stable C ABI for dynamic compute-operator plugins.
//!
//! Raw integer tags are used at the ABI boundary instead of Rust enums so a
//! plugin compiled against a newer contract cannot create an invalid Rust enum
//! discriminant in an older host. The host owns query/output structures and
//! validates their sizes before using plugin-provided fields.

use std::ffi::{c_char, c_void};

pub const VRB_OPERATOR_PLUGIN_ABI_VERSION: u32 = 1;
pub const VRB_OPERATOR_PLUGIN_ENTRY_SYMBOL: &[u8] = b"vrb_operator_plugin_entry_v1\0";
pub const VRB_OPERATOR_NAME_CAPACITY: usize = 128;
pub const VRB_OPERATOR_PLUGIN_NAME_CAPACITY: usize = 128;

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
    pub const ZERO_COPY: u64 = 1 << 0;
    pub const FP16: u64 = 1 << 1;
    pub const FP32: u64 = 1 << 2;
    pub const INT8: u64 = 1 << 3;
    pub const HOST_BYTES: u64 = 1 << 4;
    pub const EXTERNAL_RESOURCE: u64 = 1 << 5;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VrbOperatorInfoV1 {
    pub struct_size: u32,
    pub operator_id: u32,
    pub operator_kind: u32,
    pub backend_kind: u32,
    pub capability_bits: u64,
    pub name: [c_char; VRB_OPERATOR_NAME_CAPACITY],
    pub reserved: [u64; 4],
}

impl Default for VrbOperatorInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            operator_id: 0,
            operator_kind: operator_kind::CUSTOM,
            backend_kind: backend_kind::PLUGIN,
            capability_bits: 0,
            name: [0; VRB_OPERATOR_NAME_CAPACITY],
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VrbOperatorExecutionRequestV1 {
    pub struct_size: u32,
    pub operator_id: u32,
    pub flags: u64,
    pub input_ptr: *const u8,
    pub input_len: u64,
    pub output_ptr: *mut u8,
    pub output_capacity: u64,
    pub output_len: *mut u64,
    pub opaque: [u64; 4],
}

impl Default for VrbOperatorExecutionRequestV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            operator_id: 0,
            flags: 0,
            input_ptr: std::ptr::null(),
            input_len: 0,
            output_ptr: std::ptr::null_mut(),
            output_capacity: 0,
            output_len: std::ptr::null_mut(),
            opaque: [0; 4],
        }
    }
}

pub type QueryOperatorFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    index: u32,
    out_info: *mut VrbOperatorInfoV1,
) -> i32;

pub type OutputSizeFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    operator_id: u32,
    input_ptr: *const u8,
    input_len: u64,
    out_output_len: *mut u64,
) -> i32;

pub type ExecuteOperatorFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    request: *const VrbOperatorExecutionRequestV1,
) -> i32;

pub type ShutdownOperatorPluginFn = unsafe extern "C" fn(user_data: *mut c_void);

#[repr(C)]
pub struct VrbOperatorPluginV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub name: [c_char; VRB_OPERATOR_PLUGIN_NAME_CAPACITY],
    pub operator_count: u32,
    pub reserved0: u32,
    pub capability_bits: u64,
    pub user_data: *mut c_void,
    pub query_operator: Option<QueryOperatorFn>,
    pub output_size: Option<OutputSizeFn>,
    pub execute: Option<ExecuteOperatorFn>,
    pub shutdown: Option<ShutdownOperatorPluginFn>,
    pub reserved: [usize; 4],
}

pub type OperatorPluginEntryV1 = unsafe extern "C" fn() -> *const VrbOperatorPluginV1;

#[must_use]
pub const fn expected_operator_info_struct_size() -> u32 {
    std::mem::size_of::<VrbOperatorInfoV1>() as u32
}

#[must_use]
pub const fn expected_execution_request_struct_size() -> u32 {
    std::mem::size_of::<VrbOperatorExecutionRequestV1>() as u32
}

#[must_use]
pub const fn expected_operator_plugin_struct_size() -> u32 {
    std::mem::size_of::<VrbOperatorPluginV1>() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_and_struct_sizes_are_stable() {
        assert_eq!(VRB_OPERATOR_PLUGIN_ABI_VERSION, 1);
        assert!(expected_operator_info_struct_size() >= 184);
        assert!(expected_execution_request_struct_size() >= 88);
        assert!(expected_operator_plugin_struct_size() >= 224);
    }

    #[test]
    fn defaults_publish_their_actual_sizes() {
        assert_eq!(
            VrbOperatorInfoV1::default().struct_size,
            expected_operator_info_struct_size()
        );
        assert_eq!(
            VrbOperatorExecutionRequestV1::default().struct_size,
            expected_execution_request_struct_size()
        );
    }
}
