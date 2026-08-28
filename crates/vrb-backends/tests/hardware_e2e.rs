#![cfg(target_os = "windows")]

use std::ffi::c_void;
use std::time::Instant;
use vrb_backends::{run_copy_fallback_smoke, run_zero_copy_smoke, HipBackend, VulkanBackend};
use vrb_core::{BackendError, ComputeBackend};

const STRESS_ITERATIONS: u8 = 32;
const MAX_HANDLE_GROWTH: u32 = 8;

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
    fn GetProcessHandleCount(process: *mut c_void, handle_count: *mut u32) -> i32;
}

fn process_handle_count() -> u32 {
    let mut count = 0_u32;
    let process = unsafe { GetCurrentProcess() };
    let ok = unsafe { GetProcessHandleCount(process, &mut count) };
    assert_ne!(ok, 0, "GetProcessHandleCount failed");
    count
}

fn stress_bytes(iteration: u8) -> u64 {
    match iteration % 4 {
        0 => 1024 * 1024,
        1 => 4 * 1024 * 1024,
        2 => 8 * 1024 * 1024,
        _ => 16 * 1024 * 1024,
    }
}

fn median_us(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len() & 1 == 0 {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    }
}

#[test]
fn amd_vulkan_hip_bridge_full_hardware_e2e() {
    if std::env::var_os("VRB_REQUIRE_AMD_E2E").is_none() {
        return;
    }

    let vulkan = VulkanBackend::new();
    let vulkan_info = vulkan
        .runtime_info()
        .expect("Vulkan runtime discovery must succeed");
    let preferred = vulkan_info
        .preferred_compute_device()
        .expect("Vulkan must expose a preferred compute device");
    assert_eq!(
        preferred.vendor_id, 0x1002,
        "certification requires an AMD Vulkan device"
    );
    assert!(
        preferred.external_memory,
        "Vulkan external-memory support is required"
    );
    assert!(
        preferred.external_semaphore,
        "Vulkan external-semaphore support is required"
    );
    let vulkan_probe = vulkan.probe().expect("Vulkan probe must succeed");
    assert!(vulkan_probe.available);
    assert!(vulkan_probe.capabilities.external_memory);
    assert!(vulkan_probe.capabilities.external_semaphore);
    assert!(vulkan_probe.capabilities.zero_copy);

    let hip = HipBackend::new();
    let hip_info = hip
        .runtime_info()
        .expect("HIP runtime discovery must succeed");
    assert!(
        !hip_info.devices.is_empty(),
        "HIP must expose at least one device"
    );
    assert!(
        hip_info.external_memory_api,
        "HIP external-memory API is required"
    );
    assert!(
        hip_info.external_semaphore_api,
        "HIP external-semaphore API is required"
    );
    let hip_probe = hip.probe().expect("HIP probe must succeed");
    assert!(hip_probe.available);
    assert!(hip_probe.capabilities.external_memory);
    assert!(hip_probe.capabilities.external_semaphore);
    assert!(hip_probe.capabilities.zero_copy);

    let zero_error = run_zero_copy_smoke(0, 0).expect_err("zero-byte bridge request must fail");
    assert!(matches!(zero_error, BackendError::Internal(_)));
    let fallback_zero =
        run_copy_fallback_smoke(0, 0).expect_err("zero-byte copy fallback must fail");
    assert!(matches!(fallback_zero, BackendError::Internal(_)));

    // Warm up loader/runtime state before leak accounting so one-time driver caches
    // cannot be mistaken for a per-transfer resource leak.
    let warmup = run_zero_copy_smoke(1024 * 1024, 0x5a).expect("warmup bridge must succeed");
    assert_eq!(warmup.verified_bytes, 1024 * 1024);
    assert_eq!(warmup.vulkan_device, warmup.hip_device);
    let fallback_warmup =
        run_copy_fallback_smoke(1024 * 1024, 0xa5).expect("fallback warmup must succeed");
    assert_eq!(fallback_warmup.verified_bytes, 1024 * 1024);
    assert_eq!(fallback_warmup.vulkan_device, fallback_warmup.hip_device);

    let zero_handles_before = process_handle_count();
    for iteration in 0_u8..STRESS_ITERATIONS {
        let bytes = stress_bytes(iteration);
        let pattern = 0x20_u8.wrapping_add(iteration);
        let report = run_zero_copy_smoke(bytes, pattern).unwrap_or_else(|error| {
            panic!("zero-copy iteration {iteration} failed for {bytes} bytes: {error}")
        });
        assert_eq!(report.bytes, bytes);
        assert_eq!(report.verified_bytes, bytes);
        assert_eq!(report.pattern, pattern);
        assert_eq!(report.vulkan_device, report.hip_device);
        assert!(report.external_memory_handle.contains("WIN32"));
        assert!(!report.synchronization.is_empty());
    }
    let zero_handles_after = process_handle_count();
    let zero_handle_growth = zero_handles_after.saturating_sub(zero_handles_before);
    assert!(
        zero_handle_growth <= MAX_HANDLE_GROWTH,
        "zero-copy process handles grew by {zero_handle_growth} across {STRESS_ITERATIONS} iterations ({zero_handles_before} -> {zero_handles_after})"
    );

    let fallback_handles_before = process_handle_count();
    for iteration in 0_u8..STRESS_ITERATIONS {
        let bytes = stress_bytes(iteration);
        let pattern = 0x40_u8.wrapping_add(iteration);
        let report = run_copy_fallback_smoke(bytes, pattern).unwrap_or_else(|error| {
            panic!("copy-fallback iteration {iteration} failed for {bytes} bytes: {error}")
        });
        assert_eq!(report.bytes, bytes);
        assert_eq!(report.verified_bytes, bytes);
        assert_eq!(report.pattern, pattern);
        assert_eq!(report.vulkan_device, report.hip_device);
        assert!(report.transfer_path.contains("host relay"));
        assert!(!report.synchronization.is_empty());
    }
    let fallback_handles_after = process_handle_count();
    let fallback_handle_growth = fallback_handles_after.saturating_sub(fallback_handles_before);
    assert!(
        fallback_handle_growth <= MAX_HANDLE_GROWTH,
        "copy-fallback process handles grew by {fallback_handle_growth} across {STRESS_ITERATIONS} iterations ({fallback_handles_before} -> {fallback_handles_after})"
    );

    let large = run_zero_copy_smoke(64 * 1024 * 1024, 0xa5)
        .expect("final 64 MiB bridge must succeed after stress loop");
    assert_eq!(large.verified_bytes, 64 * 1024 * 1024);
    assert_eq!(large.vulkan_device, preferred.name);
    assert_eq!(large.hip_device, preferred.name);

    let fallback_large = run_copy_fallback_smoke(64 * 1024 * 1024, 0x5a)
        .expect("final 64 MiB copy fallback must succeed after stress loop");
    assert_eq!(fallback_large.verified_bytes, 64 * 1024 * 1024);
    assert_eq!(fallback_large.vulkan_device, preferred.name);
    assert_eq!(fallback_large.hip_device, preferred.name);
    assert!(fallback_large.transfer_path.contains("host relay"));

    println!(
        "VRB_STRESS_RECEIPT_JSON={{\"iterations\":{STRESS_ITERATIONS},\"max_handle_growth\":{MAX_HANDLE_GROWTH},\"zero_copy_handle_growth\":{zero_handle_growth},\"copy_fallback_handle_growth\":{fallback_handle_growth}}}"
    );

    // Criterion 10 receipt: compare two independently executed, correctness-checked
    // cross-stack transports. No synthetic delay and no claim that one must always
    // win; the measured receipt records what this exact host actually did.
    const BENCH_BYTES: u64 = 8 * 1024 * 1024;
    const BENCH_ITERATIONS: u8 = 5;
    let mut shared_us = Vec::with_capacity(BENCH_ITERATIONS as usize);
    let mut fallback_us = Vec::with_capacity(BENCH_ITERATIONS as usize);
    for iteration in 0..BENCH_ITERATIONS {
        let pattern = 0x70_u8.wrapping_add(iteration);

        let started = Instant::now();
        let shared = run_zero_copy_smoke(BENCH_BYTES, pattern)
            .expect("benchmark shared-resource transition must succeed");
        shared_us.push(started.elapsed().as_micros());
        assert_eq!(shared.verified_bytes, BENCH_BYTES);

        let started = Instant::now();
        let fallback = run_copy_fallback_smoke(BENCH_BYTES, pattern)
            .expect("benchmark host-copy fallback must succeed");
        fallback_us.push(started.elapsed().as_micros());
        assert_eq!(fallback.verified_bytes, BENCH_BYTES);
        assert!(fallback.transfer_path.contains("host relay"));
    }

    let shared_median_us = median_us(&mut shared_us);
    let fallback_median_us = median_us(&mut fallback_us);
    assert!(shared_median_us > 0);
    assert!(fallback_median_us > 0);
    let ratio = fallback_median_us as f64 / shared_median_us as f64;
    println!(
        "VRB_BENCH_RECEIPT_JSON={{\"bytes\":{BENCH_BYTES},\"iterations\":{BENCH_ITERATIONS},\"shared_median_us\":{shared_median_us},\"copy_fallback_median_us\":{fallback_median_us},\"fallback_over_shared_ratio\":{ratio:.6}}}"
    );
    println!("VRB_E2E_CERT=PASS");
}
