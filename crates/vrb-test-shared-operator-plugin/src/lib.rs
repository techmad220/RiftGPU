#![forbid(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_void};
use vrb_shared_operator_plugin_api::{
    backend_kind, capability, expected_execution_request_struct_size,
    expected_resource_region_struct_size, expected_sync_point_struct_size, memory_handle_kind,
    operator_kind, resource_access, status, sync_handle_kind, VrbSharedOperatorExecutionRequestV1,
    VrbSharedOperatorInfoV1, VrbSharedOperatorPluginV1, VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION,
    VRB_SHARED_OPERATOR_PLUGIN_NAME_CAPACITY,
};

const OPERATOR_ID: u32 = 1;
const PLUGIN_NAME: &[u8] = b"vrb-test-shared-operator";
const OPERATOR_NAME: &[u8] = b"shared-contract-validator";
const EXPECTED_METADATA: &[u8] = b"vrb-shared";
const RECEIPT: &[u8] = b"shared-contract-ok";

const fn bit(value: u32) -> u64 {
    if value < 64 {
        1_u64 << value
    } else {
        0
    }
}

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
    out_info: *mut VrbSharedOperatorInfoV1,
) -> i32 {
    if index != 0 || out_info.is_null() {
        return status::INVALID_ARGUMENT;
    }
    let info = VrbSharedOperatorInfoV1 {
        operator_id: OPERATOR_ID,
        operator_kind: operator_kind::CUSTOM,
        backend_kind: backend_kind::PLUGIN,
        capability_bits: capability::EXTERNAL_RESOURCE | capability::EXTERNAL_SYNCHRONIZATION,
        memory_kind_bits: bit(memory_handle_kind::WIN32_KMT),
        sync_kind_bits: bit(sync_handle_kind::WIN32_OPAQUE),
        name: c_name(OPERATOR_NAME),
        ..VrbSharedOperatorInfoV1::default()
    };
    // SAFETY: null was rejected above and the ABI requires a writable output
    // structure for the complete callback.
    unsafe {
        *out_info = info;
    }
    status::OK
}

unsafe extern "C" fn execute(
    _user_data: *mut c_void,
    request: *const VrbSharedOperatorExecutionRequestV1,
) -> i32 {
    if request.is_null() {
        return status::INVALID_ARGUMENT;
    }
    // SAFETY: null was rejected above and the ABI requires a readable request.
    let request = unsafe { &*request };
    if request.struct_size < expected_execution_request_struct_size()
        || request.operator_id != OPERATOR_ID
        || request.receipt_len.is_null()
    {
        return status::INVALID_ARGUMENT;
    }

    let metadata_len = match usize::try_from(request.metadata_len) {
        Ok(value) => value,
        Err(_) => return status::INVALID_ARGUMENT,
    };
    if metadata_len != EXPECTED_METADATA.len() || request.metadata_ptr.is_null() {
        return status::INVALID_ARGUMENT;
    }
    // SAFETY: pointer/length pair is supplied by the host for this callback.
    let metadata = unsafe { std::slice::from_raw_parts(request.metadata_ptr, metadata_len) };
    if metadata != EXPECTED_METADATA {
        return status::INVALID_ARGUMENT;
    }

    if request.resource_count != 1
        || request.resources_ptr.is_null()
        || request.wait_count != 1
        || request.waits_ptr.is_null()
        || request.signal_count != 1
        || request.signals_ptr.is_null()
    {
        return status::INVALID_ARGUMENT;
    }

    // SAFETY: counts are exactly one and pointers were validated non-null.
    let resource = unsafe { &*request.resources_ptr };
    if resource.struct_size < expected_resource_region_struct_size()
        || resource.handle_kind != memory_handle_kind::WIN32_KMT
        || resource.access != resource_access::READ_WRITE
        || resource.handle == 0
        || resource.length == 0
    {
        return status::INVALID_ARGUMENT;
    }
    let end = match resource.offset.checked_add(resource.length) {
        Some(value) => value,
        None => return status::INVALID_ARGUMENT,
    };
    if end > resource.allocation_size {
        return status::INVALID_ARGUMENT;
    }

    // SAFETY: counts are exactly one and pointers were validated non-null.
    let wait = unsafe { &*request.waits_ptr };
    // SAFETY: counts are exactly one and pointers were validated non-null.
    let signal = unsafe { &*request.signals_ptr };
    for point in [wait, signal] {
        if point.struct_size < expected_sync_point_struct_size()
            || point.handle_kind != sync_handle_kind::WIN32_OPAQUE
            || point.handle == 0
        {
            return status::INVALID_ARGUMENT;
        }
    }

    let required = RECEIPT.len() as u64;
    // SAFETY: receipt_len was validated non-null above.
    unsafe {
        *request.receipt_len = required;
    }
    if request.receipt_capacity < required {
        return status::BUFFER_TOO_SMALL;
    }
    if request.receipt_ptr.is_null() {
        return status::INVALID_ARGUMENT;
    }
    // SAFETY: host advertised sufficient capacity and destination is non-null.
    unsafe {
        std::ptr::copy_nonoverlapping(RECEIPT.as_ptr(), request.receipt_ptr, RECEIPT.len());
    }
    status::OK
}

#[repr(transparent)]
struct SyncPlugin(VrbSharedOperatorPluginV1);

// SAFETY: descriptor is immutable after static initialization and carries no
// mutable plugin state through user_data.
unsafe impl Sync for SyncPlugin {}

static PLUGIN: SyncPlugin = SyncPlugin(VrbSharedOperatorPluginV1 {
    abi_version: VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION,
    struct_size: std::mem::size_of::<VrbSharedOperatorPluginV1>() as u32,
    name: c_name::<VRB_SHARED_OPERATOR_PLUGIN_NAME_CAPACITY>(PLUGIN_NAME),
    operator_count: 1,
    reserved0: 0,
    capability_bits: 0,
    user_data: std::ptr::null_mut(),
    query_operator: Some(query_operator),
    execute: Some(execute),
    shutdown: None,
    reserved: [0; 5],
});

/// Return the immutable v1 shared-operator test descriptor.
///
/// # Safety
///
/// The caller must interpret the pointer according to the advertised ABI and
/// must not mutate or free the returned descriptor.
#[no_mangle]
pub unsafe extern "C" fn vrb_shared_operator_plugin_entry_v1() -> *const VrbSharedOperatorPluginV1 {
    &PLUGIN.0
}
