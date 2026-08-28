#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "windows")]
use libloading::Library;
#[cfg(target_os = "windows")]
use std::env;
use std::ffi::{c_char, c_void};
#[cfg(target_os = "windows")]
use std::path::PathBuf;
use std::ptr;
#[cfg(target_os = "windows")]
use vrb_gemm_shared_protocol::{
    decode_control, expected_resource_lengths, A_RESOURCE_INDEX, B_RESOURCE_INDEX,
    C_RESOURCE_INDEX, SHARED_GEMM_RESOURCE_COUNT,
};
use vrb_shared_operator_plugin_api::{
    backend_kind, capability, expected_execution_request_struct_size, memory_handle_kind,
    operator_kind, status, VrbSharedOperatorExecutionRequestV1, VrbSharedOperatorInfoV1,
    VrbSharedOperatorPluginV1, VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION,
    VRB_SHARED_OPERATOR_PLUGIN_NAME_CAPACITY,
};
#[cfg(target_os = "windows")]
use vrb_shared_operator_plugin_api::{
    expected_resource_region_struct_size, resource_access, VrbSharedResourceRegionV1,
};

const OPERATOR_ID: u32 = 1;
const PLUGIN_NAME: &[u8] = b"vrb-hip-shared-gemm";
const OPERATOR_NAME: &[u8] = b"hip-rocblas-shared-fp32-gemm";
#[cfg(target_os = "windows")]
const SUCCESS_RECEIPT: &[u8] = b"hip-rocblas-shared-gemm-ok";
#[cfg(target_os = "windows")]
const HIP_SUCCESS: i32 = 0;
#[cfg(target_os = "windows")]
const ROCBLAS_STATUS_SUCCESS: i32 = 0;
#[cfg(target_os = "windows")]
const ROCBLAS_OPERATION_NONE: i32 = 111;
#[cfg(target_os = "windows")]
const HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT: i32 = 3;

#[cfg(target_os = "windows")]
type HipExternalMemory = *mut c_void;
#[cfg(target_os = "windows")]
type RocblasHandle = *mut c_void;

#[cfg(target_os = "windows")]
type HipInit = unsafe extern "C" fn(u32) -> i32;
#[cfg(target_os = "windows")]
type HipSetDevice = unsafe extern "C" fn(i32) -> i32;
#[cfg(target_os = "windows")]
type HipImportExternalMemory =
    unsafe extern "C" fn(*mut HipExternalMemory, *const HipExternalMemoryHandleDesc) -> i32;
#[cfg(target_os = "windows")]
type HipExternalMemoryGetMappedBuffer = unsafe extern "C" fn(
    *mut *mut c_void,
    HipExternalMemory,
    *const HipExternalMemoryBufferDesc,
) -> i32;
#[cfg(target_os = "windows")]
type HipDestroyExternalMemory = unsafe extern "C" fn(HipExternalMemory) -> i32;
#[cfg(target_os = "windows")]
type HipFree = unsafe extern "C" fn(*mut c_void) -> i32;
#[cfg(target_os = "windows")]
type HipDeviceSynchronize = unsafe extern "C" fn() -> i32;

#[cfg(target_os = "windows")]
type RocblasCreateHandle = unsafe extern "C" fn(*mut RocblasHandle) -> i32;
#[cfg(target_os = "windows")]
type RocblasDestroyHandle = unsafe extern "C" fn(RocblasHandle) -> i32;
#[cfg(target_os = "windows")]
type RocblasSgemm = unsafe extern "C" fn(
    RocblasHandle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
) -> i32;

#[cfg(target_os = "windows")]
#[repr(C)]
#[derive(Clone, Copy)]
struct HipWin32Handle {
    handle: *mut c_void,
    name: *const c_void,
}

#[cfg(target_os = "windows")]
#[repr(C)]
union HipExternalMemoryHandleValue {
    fd: i32,
    win32: HipWin32Handle,
    nv_sci_buf_object: *const c_void,
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct HipExternalMemoryHandleDesc {
    type_: i32,
    handle: HipExternalMemoryHandleValue,
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct HipExternalMemoryBufferDesc {
    offset: u64,
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}

#[cfg(target_os = "windows")]
struct HipApi {
    _library: Library,
    set_device: HipSetDevice,
    import_external_memory: HipImportExternalMemory,
    external_memory_get_mapped_buffer: HipExternalMemoryGetMappedBuffer,
    destroy_external_memory: HipDestroyExternalMemory,
    free: HipFree,
    device_synchronize: HipDeviceSynchronize,
}

#[cfg(target_os = "windows")]
impl HipApi {
    fn load() -> Result<Self, i32> {
        let library = load_first(&hip_library_candidates()).map_err(|_| status::UNAVAILABLE)?;
        // SAFETY: symbols are resolved from the official HIP runtime and the
        // owning Library is retained by HipApi for their full lifetime.
        unsafe {
            let init = load_required::<HipInit>(&library, b"hipInit\0")?;
            let set_device = load_required::<HipSetDevice>(&library, b"hipSetDevice\0")?;
            let import_external_memory =
                load_required::<HipImportExternalMemory>(&library, b"hipImportExternalMemory\0")?;
            let external_memory_get_mapped_buffer =
                load_required::<HipExternalMemoryGetMappedBuffer>(
                    &library,
                    b"hipExternalMemoryGetMappedBuffer\0",
                )?;
            let destroy_external_memory =
                load_required::<HipDestroyExternalMemory>(&library, b"hipDestroyExternalMemory\0")?;
            let free = load_required::<HipFree>(&library, b"hipFree\0")?;
            let device_synchronize =
                load_required::<HipDeviceSynchronize>(&library, b"hipDeviceSynchronize\0")?;
            if init(0) != HIP_SUCCESS {
                return Err(status::UNAVAILABLE);
            }
            Ok(Self {
                _library: library,
                set_device,
                import_external_memory,
                external_memory_get_mapped_buffer,
                destroy_external_memory,
                free,
                device_synchronize,
            })
        }
    }

    fn import_region(&self, region: &VrbSharedResourceRegionV1) -> Result<ImportedRegion<'_>, i32> {
        let descriptor = HipExternalMemoryHandleDesc {
            type_: HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT,
            handle: HipExternalMemoryHandleValue {
                win32: HipWin32Handle {
                    handle: region.handle as usize as *mut c_void,
                    name: ptr::null(),
                },
            },
            size: region.allocation_size,
            flags: 0,
            reserved: [0; 16],
        };
        let mut external = ptr::null_mut();
        // SAFETY: the host keeps the Vulkan KMT allocation alive for the full
        // synchronous callback and descriptor layout matches HIP's public ABI.
        let import_status = unsafe { (self.import_external_memory)(&mut external, &descriptor) };
        if import_status != HIP_SUCCESS || external.is_null() {
            return Err(status::UNAVAILABLE);
        }
        let external_guard = ExternalMemoryGuard {
            api: self,
            handle: external,
        };
        let mapping = HipExternalMemoryBufferDesc {
            offset: region.offset,
            size: region.length,
            flags: 0,
            reserved: [0; 16],
        };
        let mut pointer = ptr::null_mut();
        // SAFETY: imported external memory is live and the region was already
        // validated against allocation_size before this call.
        let map_status = unsafe {
            (self.external_memory_get_mapped_buffer)(&mut pointer, external_guard.handle, &mapping)
        };
        if map_status != HIP_SUCCESS || pointer.is_null() {
            return Err(status::UNAVAILABLE);
        }
        Ok(ImportedRegion {
            mapped: MappedBufferGuard { api: self, pointer },
            external: external_guard,
        })
    }
}

#[cfg(target_os = "windows")]
struct RocblasApi {
    _library: Library,
    create_handle: RocblasCreateHandle,
    destroy_handle: RocblasDestroyHandle,
    sgemm: RocblasSgemm,
}

#[cfg(target_os = "windows")]
impl RocblasApi {
    fn load() -> Result<Self, i32> {
        let library = load_first(&rocblas_library_candidates()).map_err(|_| status::UNAVAILABLE)?;
        // SAFETY: symbols are resolved from rocBLAS and the library is retained.
        unsafe {
            Ok(Self {
                create_handle: load_required::<RocblasCreateHandle>(
                    &library,
                    b"rocblas_create_handle\0",
                )?,
                destroy_handle: load_required::<RocblasDestroyHandle>(
                    &library,
                    b"rocblas_destroy_handle\0",
                )?,
                sgemm: load_required::<RocblasSgemm>(&library, b"rocblas_sgemm\0")?,
                _library: library,
            })
        }
    }
}

#[cfg(target_os = "windows")]
struct ExternalMemoryGuard<'a> {
    api: &'a HipApi,
    handle: HipExternalMemory,
}

#[cfg(target_os = "windows")]
impl Drop for ExternalMemoryGuard<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: guard uniquely owns one HIP external-memory object.
            let _ = unsafe { (self.api.destroy_external_memory)(self.handle) };
        }
    }
}

#[cfg(target_os = "windows")]
struct MappedBufferGuard<'a> {
    api: &'a HipApi,
    pointer: *mut c_void,
}

#[cfg(target_os = "windows")]
impl Drop for MappedBufferGuard<'_> {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            // SAFETY: HIP external mapped buffers are released with hipFree.
            let _ = unsafe { (self.api.free)(self.pointer) };
        }
    }
}

#[cfg(target_os = "windows")]
struct ImportedRegion<'a> {
    // Drop mapped pointer before imported external-memory object.
    mapped: MappedBufferGuard<'a>,
    external: ExternalMemoryGuard<'a>,
}

#[cfg(target_os = "windows")]
impl ImportedRegion<'_> {
    fn pointer(&self) -> *mut c_void {
        self.mapped.pointer
    }
}

#[cfg(target_os = "windows")]
struct RocblasHandleGuard<'a> {
    api: &'a RocblasApi,
    handle: RocblasHandle,
}

#[cfg(target_os = "windows")]
impl Drop for RocblasHandleGuard<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: guard uniquely owns one rocBLAS handle.
            let _ = unsafe { (self.api.destroy_handle)(self.handle) };
        }
    }
}

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
        operator_kind: operator_kind::GEMM,
        backend_kind: backend_kind::HIP,
        capability_bits: capability::EXTERNAL_RESOURCE | capability::FP32,
        memory_kind_bits: bit(memory_handle_kind::WIN32_KMT),
        sync_kind_bits: 0,
        name: c_name(OPERATOR_NAME),
        ..VrbSharedOperatorInfoV1::default()
    };
    // SAFETY: null was rejected and caller provides writable ABI storage.
    unsafe { *out_info = info };
    status::OK
}

unsafe extern "C" fn execute(
    _user_data: *mut c_void,
    request: *const VrbSharedOperatorExecutionRequestV1,
) -> i32 {
    if request.is_null() {
        return status::INVALID_ARGUMENT;
    }
    // SAFETY: non-null request is readable for callback duration by ABI contract.
    let request = unsafe { &*request };
    if request.struct_size < expected_execution_request_struct_size()
        || request.operator_id != OPERATOR_ID
        || request.receipt_len.is_null()
    {
        return status::INVALID_ARGUMENT;
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = request;
        status::UNSUPPORTED
    }

    #[cfg(target_os = "windows")]
    {
        execute_windows(request)
    }
}

#[cfg(target_os = "windows")]
fn execute_windows(request: &VrbSharedOperatorExecutionRequestV1) -> i32 {
    let metadata = match abi_slice(request.metadata_ptr, request.metadata_len) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let control = match decode_control(metadata) {
        Ok(value) => value,
        Err(_) => return status::INVALID_ARGUMENT,
    };
    if request.resource_count as usize != SHARED_GEMM_RESOURCE_COUNT
        || request.resources_ptr.is_null()
        || request.wait_count != 0
        || request.signal_count != 0
    {
        return status::INVALID_ARGUMENT;
    }
    // SAFETY: resource pointer is non-null and count is exactly the protocol's fixed count.
    let resources =
        unsafe { std::slice::from_raw_parts(request.resources_ptr, SHARED_GEMM_RESOURCE_COUNT) };
    let lengths = match expected_resource_lengths(control) {
        Ok(value) => value,
        Err(_) => return status::INVALID_ARGUMENT,
    };
    if !validate_region(
        &resources[A_RESOURCE_INDEX],
        resource_access::READ_ONLY,
        lengths.a_bytes,
    ) || !validate_region(
        &resources[B_RESOURCE_INDEX],
        resource_access::READ_ONLY,
        lengths.b_bytes,
    ) || !validate_region(
        &resources[C_RESOURCE_INDEX],
        resource_access::READ_WRITE,
        lengths.c_bytes,
    ) {
        return status::INVALID_ARGUMENT;
    }

    let m = match i32::try_from(control.m) {
        Ok(value) => value,
        Err(_) => return status::UNSUPPORTED,
    };
    let n = match i32::try_from(control.n) {
        Ok(value) => value,
        Err(_) => return status::UNSUPPORTED,
    };
    let k = match i32::try_from(control.k) {
        Ok(value) => value,
        Err(_) => return status::UNSUPPORTED,
    };

    let hip = match HipApi::load() {
        Ok(value) => value,
        Err(error) => return error,
    };
    // SAFETY: device index was encoded by the host after correlating its Vulkan/HIP inventory.
    if unsafe { (hip.set_device)(control.hip_device_index) } != HIP_SUCCESS {
        return status::UNAVAILABLE;
    }
    let rocblas = match RocblasApi::load() {
        Ok(value) => value,
        Err(error) => return error,
    };

    let a = match hip.import_region(&resources[A_RESOURCE_INDEX]) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let b = match hip.import_region(&resources[B_RESOURCE_INDEX]) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let c = match hip.import_region(&resources[C_RESOURCE_INDEX]) {
        Ok(value) => value,
        Err(error) => return error,
    };

    let mut raw_handle = ptr::null_mut();
    // SAFETY: output points to valid local handle storage.
    if unsafe { (rocblas.create_handle)(&mut raw_handle) } != ROCBLAS_STATUS_SUCCESS
        || raw_handle.is_null()
    {
        return status::UNAVAILABLE;
    }
    let handle = RocblasHandleGuard {
        api: &rocblas,
        handle: raw_handle,
    };

    // rocBLAS is column-major. Row-major C=A*B is equivalent to the column-major
    // operation C^T=B^T*A^T on the same bytes, so no host-side transpose/copy occurs.
    // SAFETY: mapped regions have exact protocol-validated FP32 byte lengths and
    // remain live through rocBLAS execution and hipDeviceSynchronize.
    let blas_status = unsafe {
        (rocblas.sgemm)(
            handle.handle,
            ROCBLAS_OPERATION_NONE,
            ROCBLAS_OPERATION_NONE,
            n,
            m,
            k,
            &control.alpha,
            b.pointer().cast::<f32>(),
            n,
            a.pointer().cast::<f32>(),
            k,
            &control.beta,
            c.pointer().cast::<f32>(),
            n,
        )
    };
    if blas_status != ROCBLAS_STATUS_SUCCESS {
        return status::INTERNAL_ERROR;
    }
    // SAFETY: HIP runtime is live and all submitted rocBLAS work targets this device.
    if unsafe { (hip.device_synchronize)() } != HIP_SUCCESS {
        return status::INTERNAL_ERROR;
    }

    write_receipt(request, SUCCESS_RECEIPT)
}

#[cfg(target_os = "windows")]
fn validate_region(region: &VrbSharedResourceRegionV1, access: u32, expected_bytes: u64) -> bool {
    if region.struct_size < expected_resource_region_struct_size()
        || region.handle_kind != memory_handle_kind::WIN32_KMT
        || region.access != access
        || region.handle == 0
        || region.length != expected_bytes
        || region.allocation_size == 0
    {
        return false;
    }
    region
        .offset
        .checked_add(region.length)
        .is_some_and(|end| end <= region.allocation_size)
}

#[cfg(target_os = "windows")]
fn write_receipt(request: &VrbSharedOperatorExecutionRequestV1, receipt: &[u8]) -> i32 {
    let required = receipt.len() as u64;
    // SAFETY: execute validated receipt_len as non-null.
    unsafe { *request.receipt_len = required };
    if request.receipt_capacity < required {
        return status::BUFFER_TOO_SMALL;
    }
    if required > 0 && request.receipt_ptr.is_null() {
        return status::INVALID_ARGUMENT;
    }
    if !receipt.is_empty() {
        // SAFETY: host advertised sufficient capacity and destination is non-null.
        unsafe {
            ptr::copy_nonoverlapping(receipt.as_ptr(), request.receipt_ptr, receipt.len());
        }
    }
    status::OK
}

#[cfg(target_os = "windows")]
fn abi_slice<'a>(pointer: *const u8, length: u64) -> Result<&'a [u8], i32> {
    let length = usize::try_from(length).map_err(|_| status::INVALID_ARGUMENT)?;
    if length == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err(status::INVALID_ARGUMENT);
    }
    // SAFETY: ABI host promises readable memory for callback duration.
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) })
}

#[cfg(target_os = "windows")]
fn hip_library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for variable in ["HIP_PATH", "ROCM_PATH"] {
        if let Some(root) = env::var_os(variable) {
            let bin = PathBuf::from(root).join("bin");
            candidates.push(bin.join("amdhip64.dll"));
            for major in (5..=9).rev() {
                candidates.push(bin.join(format!("amdhip64_{major}.dll")));
            }
        }
    }
    candidates.push(PathBuf::from("amdhip64.dll"));
    for major in (5..=9).rev() {
        candidates.push(PathBuf::from(format!("amdhip64_{major}.dll")));
    }
    candidates
}

#[cfg(target_os = "windows")]
fn rocblas_library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for variable in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(root) = env::var_os(variable) {
            let bin = PathBuf::from(root).join("bin");
            candidates.push(bin.join("rocblas.dll"));
        }
    }
    candidates.push(PathBuf::from("rocblas.dll"));
    candidates
}

#[cfg(target_os = "windows")]
fn load_first(candidates: &[PathBuf]) -> Result<Library, ()> {
    for candidate in candidates {
        // SAFETY: candidates are restricted to administrator-selected ROCm paths
        // and official runtime library names.
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Ok(library);
        }
    }
    Err(())
}

#[cfg(target_os = "windows")]
unsafe fn load_required<T: Copy>(library: &Library, name: &[u8]) -> Result<T, i32> {
    // SAFETY: caller supplies the exact public C function-pointer type and keeps
    // the Library alive while the copied pointer is used.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|_| status::UNAVAILABLE)
}

#[repr(transparent)]
struct SyncPlugin(VrbSharedOperatorPluginV1);

// SAFETY: descriptor is immutable after initialization and user_data is null.
unsafe impl Sync for SyncPlugin {}

static PLUGIN: SyncPlugin = SyncPlugin(VrbSharedOperatorPluginV1 {
    abi_version: VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION,
    struct_size: std::mem::size_of::<VrbSharedOperatorPluginV1>() as u32,
    name: c_name::<VRB_SHARED_OPERATOR_PLUGIN_NAME_CAPACITY>(PLUGIN_NAME),
    operator_count: 1,
    reserved0: 0,
    // Deliberately no PROVEN_ZERO_COPY yet. Hardware certification must prove
    // this exact path before that evidence-bearing capability is enabled.
    capability_bits: capability::EXTERNAL_RESOURCE | capability::FP32,
    user_data: ptr::null_mut(),
    query_operator: Some(query_operator),
    execute: Some(execute),
    shutdown: None,
    reserved: [0; 5],
});

/// Return the immutable v1 shared HIP GEMM descriptor.
///
/// # Safety
/// The caller must honor `VRB_SHARED_OPERATOR_PLUGIN_ABI_VERSION` and must not
/// mutate or free the returned static descriptor.
#[no_mangle]
pub unsafe extern "C" fn vrb_shared_operator_plugin_entry_v1() -> *const VrbSharedOperatorPluginV1 {
    &PLUGIN.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_does_not_claim_zero_copy_before_hardware_certification() {
        assert_eq!(PLUGIN.0.capability_bits & capability::PROVEN_ZERO_COPY, 0);
        assert_ne!(PLUGIN.0.capability_bits & capability::EXTERNAL_RESOURCE, 0);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn hip_external_memory_abi_matches_certified_win64_layout() {
        assert_eq!(std::mem::size_of::<HipExternalMemoryHandleDesc>(), 104);
        assert_eq!(std::mem::size_of::<HipExternalMemoryBufferDesc>(), 88);
    }

    #[test]
    fn row_major_dimension_mapping_is_i32_safe_for_small_case() {
        let m = 2_i32;
        let n = 4_i32;
        let k = 3_i32;
        assert_eq!((n, m, k, n, k, n), (4, 2, 3, 4, 3, 4));
    }
}
