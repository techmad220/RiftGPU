use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::EnvFilter;
use vrb_backends::{
    run_zero_copy_smoke, CpuBackend, DynamicPluginBackend, HipBackend, VulkanBackend,
};
use vrb_core::{DataType, OperationKind, PerformanceRecord, RouteRequest, Runtime, RuntimeBuilder};

#[derive(Debug, Parser)]
#[command(
    name = "vrb",
    version,
    about = "Vulkan ROCm Bridge diagnostics and routing CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Probe Vulkan, HIP, CPU, and optional dynamic plugins.
    Doctor {
        #[arg(long)]
        json: bool,
        #[arg(long = "plugin", value_name = "PATH")]
        plugins: Vec<PathBuf>,
    },
    /// Ask the injected routing policy which backend should own an operation.
    Route {
        #[arg(value_enum)]
        operation: CliOperation,
        #[arg(value_enum)]
        data_type: CliDataType,
        #[arg(long)]
        zero_copy: bool,
        #[arg(long = "plugin", value_name = "PATH")]
        plugins: Vec<PathBuf>,
    },
    /// Run the correctness CPU vector-add microbenchmark and optionally persist its record.
    BenchCpu {
        #[arg(long, default_value_t = 1_048_576)]
        elements: usize,
        #[arg(long, default_value_t = 25)]
        iterations: u32,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Prove Vulkan and HIP can access the same GPU allocation without a CPU relay copy.
    BridgeSmoke {
        #[arg(long, default_value_t = 4_194_304)]
        bytes: u64,
        #[arg(long, default_value_t = 0x5a, value_parser = parse_byte)]
        pattern: u8,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliOperation {
    Copy,
    VectorAdd,
    Gemv,
    Gemm,
    Softmax,
    Attention,
    RmsNorm,
    Custom,
}

impl From<CliOperation> for OperationKind {
    fn from(value: CliOperation) -> Self {
        match value {
            CliOperation::Copy => Self::Copy,
            CliOperation::VectorAdd => Self::VectorAdd,
            CliOperation::Gemv => Self::Gemv,
            CliOperation::Gemm => Self::Gemm,
            CliOperation::Softmax => Self::Softmax,
            CliOperation::Attention => Self::Attention,
            CliOperation::RmsNorm => Self::RmsNorm,
            CliOperation::Custom => Self::Custom,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliDataType {
    F32,
    F16,
    Bf16,
    I8,
    Q4,
    Unknown,
}

impl From<CliDataType> for DataType {
    fn from(value: CliDataType) -> Self {
        match value {
            CliDataType::F32 => Self::F32,
            CliDataType::F16 => Self::F16,
            CliDataType::Bf16 => Self::Bf16,
            CliDataType::I8 => Self::I8,
            CliDataType::Q4 => Self::Q4,
            CliDataType::Unknown => Self::Unknown,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_target(false)
        .try_init()
        .ok();

    match Cli::parse().command {
        Command::Doctor { json, plugins } => doctor(json, &plugins),
        Command::Route {
            operation,
            data_type,
            zero_copy,
            plugins,
        } => route(operation, data_type, zero_copy, &plugins),
        Command::BenchCpu {
            elements,
            iterations,
            output,
        } => bench_cpu(elements, iterations, output),
        Command::BridgeSmoke { bytes, pattern } => bridge_smoke(bytes, pattern),
    }
}

fn build_runtime(plugins: &[PathBuf]) -> Result<Runtime> {
    let mut builder = RuntimeBuilder::new()
        .backend(Arc::new(CpuBackend::new()))?
        .backend(Arc::new(VulkanBackend::new()))?
        .backend(Arc::new(HipBackend::new()))?;

    for path in plugins {
        let plugin = DynamicPluginBackend::load(path)
            .with_context(|| format!("failed to load plugin {}", path.display()))?;
        builder = builder.backend(Arc::new(plugin))?;
    }

    Ok(builder.build())
}

fn doctor(json: bool, plugins: &[PathBuf]) -> Result<()> {
    let runtime = build_runtime(plugins)?;
    let probes = runtime.probes();

    if json {
        println!("{}", serde_json::to_string_pretty(&probes)?);
        return Ok(());
    }

    println!("Vulkan ROCm Bridge doctor");
    println!("=========================");
    for probe in probes {
        println!("{} ({:?})", probe.id, probe.kind);
        println!("  available: {}", probe.available);
        println!("  name: {}", probe.name);
        println!("  vendor: {}", probe.vendor);
        println!("  devices: {}", probe.device_count);
        println!("  external memory: {}", probe.capabilities.external_memory);
        println!(
            "  external semaphore: {}",
            probe.capabilities.external_semaphore
        );
        println!("  zero copy: {}", probe.capabilities.zero_copy);
        println!("  detail: {}", probe.detail);
    }
    Ok(())
}

fn route(
    operation: CliOperation,
    data_type: CliDataType,
    zero_copy: bool,
    plugins: &[PathBuf],
) -> Result<()> {
    let runtime = build_runtime(plugins)?;
    let mut request = RouteRequest::new(operation.into(), data_type.into());
    if zero_copy {
        request = request.zero_copy();
    }

    let selected = runtime.route(&request)?;
    println!("{selected}");
    Ok(())
}

fn bench_cpu(elements: usize, iterations: u32, output_path: Option<PathBuf>) -> Result<()> {
    if elements == 0 {
        bail!("--elements must be greater than zero");
    }
    if iterations == 0 {
        bail!("--iterations must be greater than zero");
    }

    let backend = CpuBackend::new();
    let left: Vec<f32> = (0..elements)
        .map(|index| (index % 251) as f32 * 0.25)
        .collect();
    let right: Vec<f32> = (0..elements)
        .map(|index| (index % 127) as f32 * 0.5)
        .collect();
    let mut output = vec![0.0_f32; elements];

    backend.vector_add_f32(&left, &right, &mut output)?;
    validate_vector_add(&left, &right, &output)?;

    let mut samples = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let started = Instant::now();
        backend.vector_add_f32(&left, &right, &mut output)?;
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }
    validate_vector_add(&left, &right, &output)?;

    samples.sort_by(f64::total_cmp);
    let median_microseconds = median(&samples);
    let bytes_per_iteration = elements as f64 * 3.0 * std::mem::size_of::<f32>() as f64;
    let gib_per_second =
        bytes_per_iteration / (median_microseconds / 1_000_000.0) / (1024.0 * 1024.0 * 1024.0);

    let record = PerformanceRecord {
        backend: vrb_core::BackendId::new("cpu")?,
        operation: OperationKind::VectorAdd,
        data_type: DataType::F32,
        median_microseconds,
        samples: iterations,
    };
    let json = serde_json::to_string_pretty(&record)?;

    println!("elements={elements}");
    println!("iterations={iterations}");
    println!("median_us={median_microseconds:.3}");
    println!("effective_gib_s={gib_per_second:.3}");
    println!("record={json}");

    if let Some(path) = output_path {
        std::fs::write(&path, format!("{json}\n"))
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("wrote={}", path.display());
    }

    Ok(())
}

fn bridge_smoke(bytes: u64, pattern: u8) -> Result<()> {
    let report = run_zero_copy_smoke(bytes, pattern)?;
    println!("VULKAN_DEVICE={}", report.vulkan_device);
    println!("HIP_DEVICE={}", report.hip_device);
    println!("BYTES={}", report.bytes);
    println!("PATTERN=0x{:02x}", report.pattern);
    println!("VERIFIED_BYTES={}", report.verified_bytes);
    println!("HANDLE={}", report.external_memory_handle);
    println!("SYNCHRONIZATION={}", report.synchronization);
    println!("ZERO_COPY_SMOKE=PASS");
    Ok(())
}

fn parse_byte(value: &str) -> std::result::Result<u8, String> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u8::from_str_radix(hex, 16).map_err(|error| format!("invalid byte '{value}': {error}"))
    } else {
        trimmed
            .parse::<u8>()
            .map_err(|error| format!("invalid byte '{value}': {error}"))
    }
}

fn validate_vector_add(left: &[f32], right: &[f32], output: &[f32]) -> Result<()> {
    for (index, ((left, right), actual)) in left.iter().zip(right).zip(output).enumerate() {
        let expected = *left + *right;
        if actual.to_bits() != expected.to_bits() {
            bail!("CPU reference mismatch at element {index}: expected {expected}, got {actual}");
        }
    }
    Ok(())
}

fn median(samples: &[f64]) -> f64 {
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_even_and_odd_sample_counts() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn parse_byte_accepts_decimal_and_hex() {
        assert_eq!(parse_byte("90").unwrap(), 90);
        assert_eq!(parse_byte("0x5a").unwrap(), 0x5a);
        assert!(parse_byte("0x100").is_err());
    }
}
