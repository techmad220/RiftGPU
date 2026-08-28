#![cfg(target_os = "windows")]

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;
use vrb_backends::{HipBackend, VulkanBackend};
use vrb_core::{BackendError, BackendKind};
use vrb_gemm_protocol::{decode_response, encode_request, GemmRequest};
use vrb_gemm_reference::CpuReferenceGemm;
use vrb_gemm_shared_protocol::{encode_control, SharedGemmControl};
use vrb_operators::OperatorKind;
use vrb_shared_operator_loader::LoadedSharedOperatorLibrary;
use vrb_shared_operators::{
    ExternalMemoryHandleKind, FirstCompatibleShared, ResourceAccess, SharedOperator,
    SharedOperatorInvocation, SharedOperatorRegistry, SharedOperatorRequest, SharedResourceRegion,
};
use vrb_vulkan_shared_buffer::{with_exported_shared_buffers, SharedBufferSpec};

const STRESS_ITERATIONS: u32 = 32;
const MAX_HANDLE_GROWTH: u32 = 8;

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
    fn GetProcessHandleCount(process: *mut c_void, handle_count: *mut u32) -> i32;
}

fn process_handle_count() -> u32 {
    let mut count = 0_u32;
    // SAFETY: Win32 returns a pseudo-handle valid for the current process.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: count points to writable storage.
    let ok = unsafe { GetProcessHandleCount(process, &mut count) };
    assert_ne!(ok, 0, "GetProcessHandleCount failed");
    count
}

fn normalize_device_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "FP32 readback must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact 4-byte chunk")))
        .collect()
}

fn correlated_hip_device_index() -> (i32, String) {
    let vulkan = VulkanBackend::new()
        .runtime_info()
        .expect("Vulkan runtime discovery must succeed");
    let preferred = vulkan
        .preferred_compute_device()
        .expect("Vulkan must expose a preferred compute device");
    assert_eq!(
        preferred.vendor_id, 0x1002,
        "shared GEMM certification requires an AMD Vulkan device"
    );

    let hip = HipBackend::new()
        .runtime_info()
        .expect("HIP runtime discovery must succeed");
    assert!(hip.external_memory_api, "HIP external-memory API is required");
    let target = normalize_device_name(&preferred.name);
    let (index, name) = hip
        .devices
        .iter()
        .enumerate()
        .find(|(_, name)| normalize_device_name(name) == target)
        .or_else(|| {
            hip.devices.iter().enumerate().find(|(_, name)| {
                let candidate = normalize_device_name(name);
                candidate.contains(&target) || target.contains(&candidate)
            })
        })
        .or_else(|| (hip.devices.len() == 1).then(|| (0, &hip.devices[0])))
        .expect("Vulkan GPU must correlate with a HIP device");
    (
        i32::try_from(index).expect("HIP device index must fit i32"),
        name.clone(),
    )
}

fn reference_result(
    m: u64,
    n: u64,
    k: u64,
    alpha: f32,
    beta: f32,
    a: &[f32],
    b: &[f32],
    c: &[f32],
) -> Vec<f32> {
    let request = encode_request(GemmRequest {
        m,
        n,
        k,
        alpha,
        beta,
        a,
        b,
        c: Some(c),
    })
    .expect("CPU reference request must encode");
    let response = CpuReferenceGemm::default()
        .execute_bytes(&request)
        .expect("CPU reference GEMM must execute");
    decode_response(&response)
        .expect("CPU reference response must decode")
        .values
}

struct Case<'a> {
    m: u64,
    n: u64,
    k: u64,
    alpha: f32,
    beta: f32,
    a: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    iterations: u32,
}

fn run_shared_case(
    library: &LoadedSharedOperatorLibrary,
    hip_device_index: i32,
    case: Case<'_>,
) -> (String, Vec<f32>, u32) {
    let expected = reference_result(
        case.m, case.n, case.k, case.alpha, case.beta, case.a, case.b, case.c,
    );
    let a_bytes = f32_bytes(case.a);
    let b_bytes = f32_bytes(case.b);
    let c_bytes = f32_bytes(case.c);
    let metadata = encode_control(SharedGemmControl {
        m: case.m,
        n: case.n,
        k: case.k,
        alpha: case.alpha,
        beta: case.beta,
        hip_device_index,
    })
    .expect("shared GEMM metadata must encode");

    let specs = [
        SharedBufferSpec {
            bytes: a_bytes.len() as u64,
            initial_data: Some(&a_bytes),
            readback: false,
        },
        SharedBufferSpec {
            bytes: b_bytes.len() as u64,
            initial_data: Some(&b_bytes),
            readback: false,
        },
        SharedBufferSpec {
            bytes: c_bytes.len() as u64,
            initial_data: Some(&c_bytes),
            readback: true,
        },
    ];

    let session = with_exported_shared_buffers(&specs, |exported| {
        assert_eq!(exported.len(), 3);
        let resources = [
            SharedResourceRegion {
                handle_kind: ExternalMemoryHandleKind::Win32Kmt,
                handle: exported[0].handle,
                allocation_size: exported[0].allocation_size,
                offset: 0,
                length: exported[0].logical_size,
                access: ResourceAccess::ReadOnly,
            },
            SharedResourceRegion {
                handle_kind: ExternalMemoryHandleKind::Win32Kmt,
                handle: exported[1].handle,
                allocation_size: exported[1].allocation_size,
                offset: 0,
                length: exported[1].logical_size,
                access: ResourceAccess::ReadOnly,
            },
            SharedResourceRegion {
                handle_kind: ExternalMemoryHandleKind::Win32Kmt,
                handle: exported[2].handle,
                allocation_size: exported[2].allocation_size,
                offset: 0,
                length: exported[2].logical_size,
                access: ResourceAccess::ReadWrite,
            },
        ];
        let invocation = SharedOperatorInvocation {
            metadata: &metadata,
            resources: &resources,
            waits: &[],
            signals: &[],
        };
        let request = SharedOperatorRequest {
            kind: OperatorKind::Gemm,
            preferred_backend: Some(BackendKind::Hip),
            required_memory_kind: Some(ExternalMemoryHandleKind::Win32Kmt),
            requires_synchronization: false,
            // Phase one certifies the path before the evidence-bearing bit is enabled.
            requires_proven_zero_copy: false,
        };
        let mut registry = SharedOperatorRegistry::new(Arc::new(FirstCompatibleShared));
        library.register_into(&mut registry);

        let warmup = registry
            .execute(&request, invocation.clone())
            .map_err(|error| BackendError::Internal(error.to_string()))?;
        assert_eq!(warmup.receipt, b"hip-rocblas-shared-gemm-ok");

        let handles_before = process_handle_count();
        for iteration in 0..case.iterations {
            let output = registry
                .execute(&request, invocation.clone())
                .map_err(|error| BackendError::Internal(format!(
                    "shared GEMM iteration {iteration} failed: {error}"
                )))?;
            assert_eq!(output.receipt, b"hip-rocblas-shared-gemm-ok");
        }
        let handles_after = process_handle_count();
        Ok(handles_after.saturating_sub(handles_before))
    })
    .expect("Vulkan-export -> HIP/rocBLAS -> Vulkan-reacquire session must succeed");

    let readback = session.readbacks[2]
        .as_deref()
        .expect("C buffer must be read back");
    let actual = decode_f32_bytes(readback);
    assert_eq!(actual, expected, "HIP rocBLAS result must match CPU oracle");
    (session.vulkan_device, actual, session.value)
}

#[test]
fn amd_vulkan_to_hip_rocblas_shared_gemm_matches_cpu_oracle() {
    if std::env::var_os("VRB_REQUIRE_AMD_SHARED_GEMM_E2E").is_none() {
        return;
    }

    let path = std::env::var_os("VRB_HIP_SHARED_GEMM_PLUGIN_PATH")
        .map(PathBuf::from)
        .expect("VRB_HIP_SHARED_GEMM_PLUGIN_PATH is required for hardware certification");
    assert!(path.is_file(), "HIP shared GEMM plugin is missing: {}", path.display());
    let library = LoadedSharedOperatorLibrary::load(&path)
        .expect("HIP shared GEMM plugin must load");
    let capabilities = library.operators()[0].capabilities();
    assert!(!capabilities.proven_zero_copy, "phase-one build must not pre-claim proof");

    let (hip_device_index, hip_device_name) = correlated_hip_device_index();
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = [0.0_f32; 4];
    let (vulkan_device, product, handle_growth) = run_shared_case(
        &library,
        hip_device_index,
        Case {
            m: 2,
            n: 2,
            k: 3,
            alpha: 1.0,
            beta: 0.0,
            a: &a,
            b: &b,
            c: &c,
            iterations: STRESS_ITERATIONS,
        },
    );
    assert_eq!(product, [58.0, 64.0, 139.0, 154.0]);
    assert_eq!(normalize_device_name(&vulkan_device), normalize_device_name(&hip_device_name));
    assert!(
        handle_growth <= MAX_HANDLE_GROWTH,
        "process handles grew by {handle_growth} across {STRESS_ITERATIONS} shared GEMM iterations"
    );

    let a2 = [2.0_f32, 4.0];
    let b2 = [1.0_f32, 3.0, 2.0, 5.0];
    let c2 = [10.0_f32, 20.0];
    let (_, alpha_beta_result, _) = run_shared_case(
        &library,
        hip_device_index,
        Case {
            m: 1,
            n: 2,
            k: 2,
            alpha: 0.5,
            beta: 2.0,
            a: &a2,
            b: &b2,
            c: &c2,
            iterations: 1,
        },
    );
    assert_eq!(alpha_beta_result, [25.0, 53.0]);

    println!(
        "VRB_SHARED_GEMM_CERT_JSON={{\"iterations\":{STRESS_ITERATIONS},\"max_handle_growth\":{MAX_HANDLE_GROWTH},\"handle_growth\":{handle_growth},\"vulkan_device\":\"{vulkan_device}\",\"hip_device\":\"{hip_device_name}\"}}"
    );
    println!("VRB_SHARED_GEMM_CERT=PASS");
}
