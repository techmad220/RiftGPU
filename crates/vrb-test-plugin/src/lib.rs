#![forbid(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_void};
use vrb_plugin_api::{
    capability, VrbBackendInfoV1, VrbBackendKind, VrbExecutionRequestV1, VrbPluginV1, VrbStatus,
    VRB_PLUGIN_ABI_VERSION,
};

const NAME: &[u8] = b"certification-fixture\0";

#[repr(transparent)]
struct SyncPlugin(VrbPluginV1);

unsafe impl Sync for SyncPlugin {}

unsafe extern "C" fn probe(_user_data: *mut c_void, out_info: *mut VrbBackendInfoV1) -> VrbStatus {
    if out_info.is_null() {
        return VrbStatus::InvalidArgument;
    }

    let mut info = VrbBackendInfoV1::default();
    info.backend_kind = VrbBackendKind::Hybrid;
    info.capability_bits = capability::COMPUTE
        | capability::FP32
        | capability::EXTERNAL_MEMORY
        | capability::EXTERNAL_SEMAPHORE
        | capability::ZERO_COPY;
    info.device_count = 1;
    write_c_string(&mut info.name, b"VRB E2E Fixture");
    write_c_string(&mut info.vendor, b"Techmad Certification");

    unsafe {
        *out_info = info;
    }
    VrbStatus::Ok
}

unsafe extern "C" fn execute(
    _user_data: *mut c_void,
    request: *const VrbExecutionRequestV1,
) -> VrbStatus {
    if request.is_null() {
        return VrbStatus::InvalidArgument;
    }

    let request = unsafe { &*request };
    if request.struct_size < std::mem::size_of::<VrbExecutionRequestV1>() as u32 {
        return VrbStatus::InvalidArgument;
    }

    match request.operation {
        42 => VrbStatus::Ok,
        7 => VrbStatus::Unsupported,
        _ => VrbStatus::InvalidArgument,
    }
}

unsafe extern "C" fn shutdown(_user_data: *mut c_void) {
    if let Ok(path) = std::env::var("VRB_TEST_PLUGIN_SHUTDOWN_MARKER") {
        let _ = std::fs::write(path, b"shutdown-called\n");
    }
}

static PLUGIN: SyncPlugin = SyncPlugin(VrbPluginV1 {
    abi_version: VRB_PLUGIN_ABI_VERSION,
    struct_size: std::mem::size_of::<VrbPluginV1>() as u32,
    name: NAME.as_ptr().cast::<c_char>(),
    backend_kind: VrbBackendKind::Hybrid,
    capability_bits: capability::COMPUTE
        | capability::FP32
        | capability::EXTERNAL_MEMORY
        | capability::EXTERNAL_SEMAPHORE
        | capability::ZERO_COPY,
    user_data: std::ptr::null_mut(),
    probe: Some(probe),
    execute: Some(execute),
    shutdown: Some(shutdown),
});

#[no_mangle]
pub unsafe extern "C" fn vrb_plugin_entry_v1() -> *const VrbPluginV1 {
    &PLUGIN.0
}

fn write_c_string<const N: usize>(output: &mut [c_char; N], value: &[u8]) {
    let copy_len = value.len().min(N.saturating_sub(1));
    for (slot, byte) in output.iter_mut().zip(value.iter()).take(copy_len) {
        *slot = *byte as c_char;
    }
    if N > 0 {
        output[copy_len] = 0;
    }
}
