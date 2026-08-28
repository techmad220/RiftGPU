#![forbid(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_void};
use vrb_operator_plugin_api::{
    backend_kind, capability, operator_kind, status, VrbOperatorExecutionRequestV1,
    VrbOperatorInfoV1, VrbOperatorPluginV1, VRB_OPERATOR_PLUGIN_ABI_VERSION,
    VRB_OPERATOR_PLUGIN_NAME_CAPACITY,
};

const FIXTURE_OPERATOR_ID: u32 = 7;

const fn c_name<const N: usize>(value: &[u8]) -> [c_char; N] {
    let mut output = [0; N];
    let mut index = 0;
    while index < value.len() && index + 1 < N {
        output[index] = value[index] as c_char;
        index += 1;
    }
    output
}

unsafe extern "C" fn query_operator(
    _user_data: *mut c_void,
    index: u32,
    out_info: *mut VrbOperatorInfoV1,
) -> i32 {
    if out_info.is_null() || index != 0 {
        return status::INVALID_ARGUMENT;
    }

    let info = VrbOperatorInfoV1 {
        operator_id: FIXTURE_OPERATOR_ID,
        operator_kind: operator_kind::GEMM,
        backend_kind: backend_kind::CPU,
        capability_bits: capability::FP32 | capability::HOST_BYTES,
        name: c_name(b"fixture-gemm"),
        ..VrbOperatorInfoV1::default()
    };

    // SAFETY: null was rejected and the host contract requires a writable
    // VrbOperatorInfoV1 for the duration of the callback.
    unsafe {
        *out_info = info;
    }
    status::OK
}

unsafe extern "C" fn output_size(
    _user_data: *mut c_void,
    operator_id: u32,
    _input_ptr: *const u8,
    input_len: u64,
    out_output_len: *mut u64,
) -> i32 {
    if operator_id != FIXTURE_OPERATOR_ID || out_output_len.is_null() {
        return status::INVALID_ARGUMENT;
    }

    // SAFETY: null was rejected and the host contract requires a writable u64.
    unsafe {
        *out_output_len = input_len;
    }
    status::OK
}

unsafe extern "C" fn execute(
    _user_data: *mut c_void,
    request: *const VrbOperatorExecutionRequestV1,
) -> i32 {
    if request.is_null() {
        return status::INVALID_ARGUMENT;
    }

    // SAFETY: null was rejected and the host contract requires a readable
    // VrbOperatorExecutionRequestV1 for the duration of the callback.
    let request = unsafe { &*request };
    if request.struct_size < std::mem::size_of::<VrbOperatorExecutionRequestV1>() as u32
        || request.operator_id != FIXTURE_OPERATOR_ID
        || request.output_len.is_null()
    {
        return status::INVALID_ARGUMENT;
    }
    if request.input_len > request.output_capacity {
        return status::BUFFER_TOO_SMALL;
    }
    if request.input_len > 0 && (request.input_ptr.is_null() || request.output_ptr.is_null()) {
        return status::INVALID_ARGUMENT;
    }

    let length = match usize::try_from(request.input_len) {
        Ok(length) => length,
        Err(_) => return status::INVALID_ARGUMENT,
    };
    if length > 0 {
        // SAFETY: the host guarantees readable input and writable output buffers
        // of at least input_len/output_capacity bytes. The regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(request.input_ptr, request.output_ptr, length);
        }
    }
    // SAFETY: output_len was validated non-null above.
    unsafe {
        *request.output_len = request.input_len;
    }
    status::OK
}

unsafe extern "C" fn shutdown(_user_data: *mut c_void) {
    if let Ok(path) = std::env::var("VRB_TEST_OPERATOR_PLUGIN_SHUTDOWN_MARKER") {
        let _ = std::fs::write(path, b"shutdown-called\n");
    }
}

#[repr(transparent)]
struct SyncPlugin(VrbOperatorPluginV1);

// SAFETY: the static descriptor is immutable after initialization. The fixture
// callbacks do not mutate shared user_data, which is null.
unsafe impl Sync for SyncPlugin {}

static PLUGIN: SyncPlugin = SyncPlugin(VrbOperatorPluginV1 {
    abi_version: VRB_OPERATOR_PLUGIN_ABI_VERSION,
    struct_size: std::mem::size_of::<VrbOperatorPluginV1>() as u32,
    name: c_name::<VRB_OPERATOR_PLUGIN_NAME_CAPACITY>(b"operator-certification-fixture"),
    operator_count: 1,
    reserved0: 0,
    capability_bits: capability::HOST_BYTES,
    user_data: std::ptr::null_mut(),
    query_operator: Some(query_operator),
    output_size: Some(output_size),
    execute: Some(execute),
    shutdown: Some(shutdown),
    reserved: [0; 4],
});

/// Return the immutable v1 operator-plugin descriptor.
///
/// # Safety
///
/// The caller must interpret the returned pointer according to the
/// `VRB_OPERATOR_PLUGIN_ABI_VERSION` C ABI and must not mutate or free it.
#[no_mangle]
pub unsafe extern "C" fn vrb_operator_plugin_entry_v1() -> *const VrbOperatorPluginV1 {
    &PLUGIN.0
}
