#![forbid(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_void};

pub const VRB_PLUGIN_ABI_VERSION: u32 = 1;
pub const VRB_PLUGIN_ENTRY_SYMBOL: &[u8] = b"vrb_plugin_entry_v1\0";

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrbStatus {
    Ok = 0,
    InvalidArgument = 1,
    Unsupported = 2,
    Unavailable = 3,
    InternalError = 4,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrbBackendKind {
    Cpu = 1,
    Vulkan = 2,
    Hip = 3,
    Hybrid = 4,
    Other = 255,
}

pub mod capability {
    pub const COMPUTE: u64 = 1 << 0;
    pub const EXTERNAL_MEMORY: u64 = 1 << 1;
    pub const EXTERNAL_SEMAPHORE: u64 = 1 << 2;
    pub const ZERO_COPY: u64 = 1 << 3;
    pub const TIMESTAMPS: u64 = 1 << 4;
    pub const FP16: u64 = 1 << 5;
    pub const FP32: u64 = 1 << 6;
    pub const INT8: u64 = 1 << 7;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VrbBackendInfoV1 {
    pub struct_size: u32,
    pub backend_kind: VrbBackendKind,
    pub capability_bits: u64,
    pub device_count: u32,
    pub reserved: u32,
    pub name: [c_char; 128],
    pub vendor: [c_char; 64],
}

impl Default for VrbBackendInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            backend_kind: VrbBackendKind::Other,
            capability_bits: 0,
            device_count: 0,
            reserved: 0,
            name: [0; 128],
            vendor: [0; 64],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VrbExecutionRequestV1 {
    pub struct_size: u32,
    pub operation: u32,
    pub data_type: u32,
    pub flags: u64,
    pub element_count: u64,
    pub input_handle: u64,
    pub output_handle: u64,
    pub opaque: u64,
}

pub type ProbeFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    out_info: *mut VrbBackendInfoV1,
) -> VrbStatus;

pub type ExecuteFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    request: *const VrbExecutionRequestV1,
) -> VrbStatus;

pub type ShutdownFn = unsafe extern "C" fn(user_data: *mut c_void);

#[repr(C)]
pub struct VrbPluginV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub name: *const c_char,
    pub backend_kind: VrbBackendKind,
    pub capability_bits: u64,
    pub user_data: *mut c_void,
    pub probe: Option<ProbeFn>,
    pub execute: Option<ExecuteFn>,
    pub shutdown: Option<ShutdownFn>,
}

pub type PluginEntryV1 = unsafe extern "C" fn() -> *const VrbPluginV1;

pub const fn expected_plugin_struct_size() -> u32 {
    std::mem::size_of::<VrbPluginV1>() as u32
}

pub const fn expected_backend_info_struct_size() -> u32 {
    std::mem::size_of::<VrbBackendInfoV1>() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(VRB_PLUGIN_ABI_VERSION, 1);
        assert!(expected_plugin_struct_size() >= 48);
        assert!(expected_backend_info_struct_size() >= 200);
    }
}
