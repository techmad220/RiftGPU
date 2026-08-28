#![forbid(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_void};
use vrb_gemm_reference::{
    execute_gemm_bytes, required_output_len, GemmLimits, ReferenceGemmError, REFERENCE_GEMM_NAME,
};
use vrb_operator_plugin_api::{
    backend_kind, capability, operator_kind, status, VrbOperatorExecutionRequestV1,
    VrbOperatorInfoV1, VrbOperatorPluginV1, VRB_OPERATOR_PLUGIN_ABI_VERSION,
    VRB_OPERATOR_PLUGIN_NAME_CAPACITY,
};

const REFERENCE_GEMM_OPERATOR_ID: u32 = 1;
const PLUGIN_NAME: &[u8] = b"vrb-reference-gemm";

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
    if index != 0 || out_info.is_null() {
        return status::INVALID_ARGUMENT;
    }

    let info = VrbOperatorInfoV1 {
        operator_id: REFERENCE_GEMM_OPERATOR_ID,
        operator_kind: operator_kind::GEMM,
        backend_kind: backend_kind::CPU,
        capability_bits: capability::FP32 | capability::HOST_BYTES,
        name: c_name(REFERENCE_GEMM_NAME.as_bytes()),
        ..VrbOperatorInfoV1::default()
    };

    // SAFETY: null was rejected above and the ABI contract requires a writable
    // VrbOperatorInfoV1 for the duration of this callback.
    unsafe {
        *out_info = info;
    }
    status::OK
}

unsafe extern "C" fn output_size(
    _user_data: *mut c_void,
    operator_id: u32,
    input_ptr: *const u8,
    input_len: u64,
    out_output_len: *mut u64,
) -> i32 {
    if operator_id != REFERENCE_GEMM_OPERATOR_ID || out_output_len.is_null() {
        return status::INVALID_ARGUMENT;
    }

    let input = match unsafe { abi_input(input_ptr, input_len) } {
        Ok(input) => input,
        Err(status) => return status,
    };
    match required_output_len(input, GemmLimits::default()) {
        Ok(length) => {
            // SAFETY: null was rejected above and the ABI contract requires a
            // writable u64 for the duration of the callback.
            unsafe {
                *out_output_len = length;
            }
            status::OK
        }
        Err(error) => map_error(&error),
    }
}

unsafe extern "C" fn execute(
    _user_data: *mut c_void,
    request: *const VrbOperatorExecutionRequestV1,
) -> i32 {
    if request.is_null() {
        return status::INVALID_ARGUMENT;
    }

    // SAFETY: null was rejected above and the ABI requires a readable request
    // structure for the duration of this callback.
    let request = unsafe { &*request };
    if request.struct_size < std::mem::size_of::<VrbOperatorExecutionRequestV1>() as u32
        || request.operator_id != REFERENCE_GEMM_OPERATOR_ID
        || request.output_len.is_null()
    {
        return status::INVALID_ARGUMENT;
    }

    let input = match unsafe { abi_input(request.input_ptr, request.input_len) } {
        Ok(input) => input,
        Err(status) => return status,
    };
    let output = match execute_gemm_bytes(input, GemmLimits::default()) {
        Ok(output) => output,
        Err(error) => return map_error(&error),
    };
    let required = match u64::try_from(output.len()) {
        Ok(required) => required,
        Err(_) => return status::INTERNAL_ERROR,
    };

    // SAFETY: output_len was validated non-null above.
    unsafe {
        *request.output_len = required;
    }
    if required > request.output_capacity {
        return status::BUFFER_TOO_SMALL;
    }
    if !output.is_empty() && request.output_ptr.is_null() {
        return status::INVALID_ARGUMENT;
    }

    if !output.is_empty() {
        // SAFETY: the host provided output_capacity >= output.len() and a
        // non-null destination. The source Vec and host destination do not
        // overlap because the plugin allocated the source independently.
        unsafe {
            std::ptr::copy_nonoverlapping(output.as_ptr(), request.output_ptr, output.len());
        }
    }
    status::OK
}

unsafe fn abi_input<'a>(input_ptr: *const u8, input_len: u64) -> Result<&'a [u8], i32> {
    let length = usize::try_from(input_len).map_err(|_| status::INVALID_ARGUMENT)?;
    if length == 0 {
        return Ok(&[]);
    }
    if input_ptr.is_null() {
        return Err(status::INVALID_ARGUMENT);
    }

    // SAFETY: the ABI caller promises a readable input buffer of input_len
    // bytes for the duration of the callback, and null was rejected above.
    Ok(unsafe { std::slice::from_raw_parts(input_ptr, length) })
}

fn map_error(error: &ReferenceGemmError) -> i32 {
    match error {
        ReferenceGemmError::Protocol(_) => status::INVALID_ARGUMENT,
        ReferenceGemmError::ResourceLimit { .. } | ReferenceGemmError::AddressSpaceOverflow => {
            status::UNSUPPORTED
        }
    }
}

#[repr(transparent)]
struct SyncPlugin(VrbOperatorPluginV1);

// SAFETY: the descriptor is immutable after static initialization and this
// plugin has no mutable user_data state.
unsafe impl Sync for SyncPlugin {}

static PLUGIN: SyncPlugin = SyncPlugin(VrbOperatorPluginV1 {
    abi_version: VRB_OPERATOR_PLUGIN_ABI_VERSION,
    struct_size: std::mem::size_of::<VrbOperatorPluginV1>() as u32,
    name: c_name::<VRB_OPERATOR_PLUGIN_NAME_CAPACITY>(PLUGIN_NAME),
    operator_count: 1,
    reserved0: 0,
    capability_bits: capability::FP32 | capability::HOST_BYTES,
    user_data: std::ptr::null_mut(),
    query_operator: Some(query_operator),
    output_size: Some(output_size),
    execute: Some(execute),
    shutdown: None,
    reserved: [0; 4],
});

/// Return the immutable v1 CPU reference GEMM plugin descriptor.
///
/// # Safety
///
/// The caller must interpret the returned pointer according to the
/// `VRB_OPERATOR_PLUGIN_ABI_VERSION` contract and must not mutate or free it.
#[no_mangle]
pub unsafe extern "C" fn vrb_operator_plugin_entry_v1() -> *const VrbOperatorPluginV1 {
    &PLUGIN.0
}
