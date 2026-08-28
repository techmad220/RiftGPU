use libloading::Library;
use std::env;
use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;
use vrb_core::{
    BackendError, BackendId, BackendKind, BackendProbe, CapabilitySet, ComputeBackend, DataType,
    OperationKind,
};

type HipError = i32;
type HipDevice = i32;
type HipInit = unsafe extern "C" fn(flags: u32) -> HipError;
type HipGetDeviceCount = unsafe extern "C" fn(count: *mut i32) -> HipError;
type HipDeviceGetName = unsafe extern "C" fn(name: *mut c_char, len: i32, device: HipDevice) -> HipError;
type HipRuntimeGetVersion = unsafe extern "C" fn(version: *mut i32) -> HipError;

#[derive(Debug, Clone)]
pub struct HipRuntimeInfo {
    pub library: String,
    pub runtime_version_raw: i32,
    pub devices: Vec<String>,
    pub external_memory_api: bool,
    pub external_semaphore_api: bool,
}

#[derive(Debug)]
pub struct HipBackend {
    id: BackendId,
}

impl HipBackend {
    pub fn new() -> Self {
        Self {
            id: BackendId::new("hip").expect("static backend id is valid"),
        }
    }

    pub fn runtime_info(&self) -> Result<HipRuntimeInfo, BackendError> {
        let (library, loaded_from) = load_hip_library()?;

        // SAFETY: every symbol is looked up by the public HIP C ABI name. We call
        // only functions whose signatures are stable in the HIP runtime API and
        // keep the Library alive for the complete lifetime of all Symbol values.
        unsafe {
            let hip_init = library
                .get::<HipInit>(b"hipInit\0")
                .map_err(|error| BackendError::Probe(format!("hipInit missing: {error}")))?;
            let hip_get_device_count = library
                .get::<HipGetDeviceCount>(b"hipGetDeviceCount\0")
                .map_err(|error| BackendError::Probe(format!("hipGetDeviceCount missing: {error}")))?;
            let hip_device_get_name = library
                .get::<HipDeviceGetName>(b"hipDeviceGetName\0")
                .map_err(|error| BackendError::Probe(format!("hipDeviceGetName missing: {error}")))?;
            let hip_runtime_get_version = library
                .get::<HipRuntimeGetVersion>(b"hipRuntimeGetVersion\0")
                .map_err(|error| BackendError::Probe(format!("hipRuntimeGetVersion missing: {error}")))?;

            check_hip(hip_init(0), "hipInit")?;

            let mut runtime_version = 0_i32;
            check_hip(
                hip_runtime_get_version(&mut runtime_version),
                "hipRuntimeGetVersion",
            )?;

            let mut device_count = 0_i32;
            check_hip(hip_get_device_count(&mut device_count), "hipGetDeviceCount")?;
            if device_count < 0 {
                return Err(BackendError::Probe(
                    "HIP returned a negative device count".to_owned(),
                ));
            }

            let mut devices = Vec::with_capacity(device_count as usize);
            for index in 0..device_count {
                let mut buffer = [0_i8; 256];
                let result = hip_device_get_name(buffer.as_mut_ptr(), buffer.len() as i32, index);
                if result == 0 {
                    let name = CStr::from_ptr(buffer.as_ptr()).to_string_lossy().into_owned();
                    devices.push(if name.is_empty() {
                        format!("HIP device {index}")
                    } else {
                        name
                    });
                } else {
                    devices.push(format!("HIP device {index} (name error {result})"));
                }
            }

            let external_memory_api = symbol_exists(&library, b"hipImportExternalMemory\0")
                && symbol_exists(&library, b"hipExternalMemoryGetMappedBuffer\0")
                && symbol_exists(&library, b"hipDestroyExternalMemory\0");
            let external_semaphore_api = symbol_exists(&library, b"hipImportExternalSemaphore\0")
                && symbol_exists(&library, b"hipSignalExternalSemaphoresAsync\0")
                && symbol_exists(&library, b"hipWaitExternalSemaphoresAsync\0")
                && symbol_exists(&library, b"hipDestroyExternalSemaphore\0");

            Ok(HipRuntimeInfo {
                library: loaded_from,
                runtime_version_raw: runtime_version,
                devices,
                external_memory_api,
                external_semaphore_api,
            })
        }
    }
}

impl Default for HipBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for HipBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Hip
    }

    fn probe(&self) -> Result<BackendProbe, BackendError> {
        let info = self.runtime_info()?;
        let external_ready = info.external_memory_api && info.external_semaphore_api;
        let available = !info.devices.is_empty();

        Ok(BackendProbe {
            id: self.id.clone(),
            kind: BackendKind::Hip,
            name: info
                .devices
                .first()
                .cloned()
                .unwrap_or_else(|| "HIP runtime".to_owned()),
            vendor: "AMD ROCm/HIP".to_owned(),
            available,
            device_count: info.devices.len() as u32,
            detail: format!(
                "library={}, runtime_version_raw={}, external_memory={}, external_semaphore={}",
                info.library,
                info.runtime_version_raw,
                info.external_memory_api,
                info.external_semaphore_api
            ),
            capabilities: CapabilitySet {
                // The built-in HIP layer is deliberately a transport/runtime layer.
                // Operator plugins advertise GEMM/attention/etc. once loaded.
                operations: vec![OperationKind::Custom],
                data_types: vec![DataType::Unknown],
                external_memory: info.external_memory_api,
                external_semaphore: info.external_semaphore_api,
                zero_copy: external_ready,
            },
        })
    }
}

fn check_hip(code: HipError, operation: &str) -> Result<(), BackendError> {
    if code == 0 {
        Ok(())
    } else {
        Err(BackendError::Unavailable(format!(
            "{operation} returned HIP error code {code}"
        )))
    }
}

fn candidate_libraries() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    #[cfg(target_os = "windows")]
    {
        for variable in ["HIP_PATH", "ROCM_PATH"] {
            if let Some(root) = env::var_os(variable) {
                candidates.push(PathBuf::from(root).join("bin").join("amdhip64.dll"));
            }
        }
        candidates.push(PathBuf::from("amdhip64.dll"));
    }

    #[cfg(target_os = "linux")]
    {
        for variable in ["HIP_PATH", "ROCM_PATH"] {
            if let Some(root) = env::var_os(variable) {
                candidates.push(PathBuf::from(root).join("lib").join("libamdhip64.so"));
                candidates.push(PathBuf::from(root).join("lib64").join("libamdhip64.so"));
            }
        }
        candidates.push(PathBuf::from("/opt/rocm/lib/libamdhip64.so"));
        candidates.push(PathBuf::from("libamdhip64.so"));
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        candidates.push(PathBuf::from("libamdhip64"));
    }

    candidates
}

fn load_hip_library() -> Result<(Library, String), BackendError> {
    let mut failures = Vec::new();
    for candidate in candidate_libraries() {
        // SAFETY: loading a dynamic library is inherently unsafe because library
        // initializers may execute. The candidates are restricted to the official
        // HIP runtime names/locations selected by the local administrator.
        match unsafe { Library::new(&candidate) } {
            Ok(library) => return Ok((library, candidate.display().to_string())),
            Err(error) => failures.push(format!("{}: {error}", candidate.display())),
        }
    }

    Err(BackendError::Unavailable(format!(
        "HIP runtime library was not loadable ({})",
        failures.join("; ")
    )))
}

unsafe fn symbol_exists(library: &Library, name: &[u8]) -> bool {
    // SAFETY: the returned address is never called or dereferenced. The lookup is
    // only used to verify that a named public HIP API entry point is exported.
    unsafe { library.get::<*const c_void>(name).is_ok() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_candidates_are_non_empty() {
        assert!(!candidate_libraries().is_empty());
    }

    #[test]
    fn hip_error_zero_is_success() {
        assert!(check_hip(0, "test").is_ok());
        assert!(check_hip(1, "test").is_err());
    }
}
