use vrb_core::BackendError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroCopySmokeReport {
    pub vulkan_device: String,
    pub hip_device: String,
    pub bytes: u64,
    pub pattern: u8,
    pub verified_bytes: u64,
    pub external_memory_handle: &'static str,
    pub synchronization: &'static str,
}

#[cfg(target_os = "windows")]
pub fn run_zero_copy_smoke(bytes: u64, pattern: u8) -> Result<ZeroCopySmokeReport, BackendError> {
    windows::run(bytes, pattern)
}

#[cfg(not(target_os = "windows"))]
pub fn run_zero_copy_smoke(_bytes: u64, _pattern: u8) -> Result<ZeroCopySmokeReport, BackendError> {
    Err(BackendError::Unsupported(
        "the v0.1 zero-copy smoke test currently implements the Windows Win32/KMT path only"
            .to_owned(),
    ))
}

#[cfg(target_os = "windows")]
mod windows {
    use super::ZeroCopySmokeReport;
    use ash::{khr, vk, Device, Entry, Instance};
    use libloading::Library;
    use std::env;
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::path::PathBuf;
    use std::ptr;
    use std::slice;
    use vrb_core::BackendError;

    const AMD_PCI_VENDOR_ID: u32 = 0x1002;
    const HIP_SUCCESS: i32 = 0;
    const HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT: i32 = 3;
    const DEFAULT_MAX_SMOKE_BYTES: u64 = 256 * 1024 * 1024;

    type HipError = i32;
    type HipExternalMemory = *mut c_void;
    type HipInit = unsafe extern "C" fn(u32) -> HipError;
    type HipGetDeviceCount = unsafe extern "C" fn(*mut i32) -> HipError;
    type HipDeviceGetName = unsafe extern "C" fn(*mut c_char, i32, i32) -> HipError;
    type HipSetDevice = unsafe extern "C" fn(i32) -> HipError;
    type HipImportExternalMemory = unsafe extern "C" fn(
        *mut HipExternalMemory,
        *const HipExternalMemoryHandleDesc,
    ) -> HipError;
    type HipExternalMemoryGetMappedBuffer = unsafe extern "C" fn(
        *mut *mut c_void,
        HipExternalMemory,
        *const HipExternalMemoryBufferDesc,
    ) -> HipError;
    type HipDestroyExternalMemory = unsafe extern "C" fn(HipExternalMemory) -> HipError;
    type HipMemset = unsafe extern "C" fn(*mut c_void, i32, usize) -> HipError;
    type HipDeviceSynchronize = unsafe extern "C" fn() -> HipError;
    type HipGetErrorString = unsafe extern "C" fn(HipError) -> *const c_char;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct HipWin32Handle {
        handle: *mut c_void,
        name: *const c_void,
    }

    #[repr(C)]
    union HipExternalMemoryHandleValue {
        fd: i32,
        win32: HipWin32Handle,
        nv_sci_buf_object: *const c_void,
    }

    #[repr(C)]
    struct HipExternalMemoryHandleDesc {
        type_: i32,
        handle: HipExternalMemoryHandleValue,
        size: u64,
        flags: u32,
        reserved: [u32; 16],
    }

    #[repr(C)]
    struct HipExternalMemoryBufferDesc {
        offset: u64,
        size: u64,
        flags: u32,
        reserved: [u32; 16],
    }

    struct HipRuntime {
        _library: Library,

        get_device_count: HipGetDeviceCount,
        device_get_name: HipDeviceGetName,
        set_device: HipSetDevice,
        import_external_memory: HipImportExternalMemory,
        external_memory_get_mapped_buffer: HipExternalMemoryGetMappedBuffer,
        destroy_external_memory: HipDestroyExternalMemory,
        memset: HipMemset,
        device_synchronize: HipDeviceSynchronize,
        get_error_string: Option<HipGetErrorString>,
    }

    impl HipRuntime {
        fn load() -> Result<Self, BackendError> {
            let (library, _library_path) = load_hip_library()?;

            // SAFETY: all symbols are resolved from the loaded HIP runtime using
            // the public HIP C API names. Function pointers are copied while the
            // Library itself is retained by HipRuntime for their entire lifetime.
            unsafe {
                let init = load_required::<HipInit>(&library, b"hipInit\0")?;
                let get_device_count =
                    load_required::<HipGetDeviceCount>(&library, b"hipGetDeviceCount\0")?;
                let device_get_name =
                    load_required::<HipDeviceGetName>(&library, b"hipDeviceGetName\0")?;
                let set_device = load_required::<HipSetDevice>(&library, b"hipSetDevice\0")?;
                let import_external_memory = load_required::<HipImportExternalMemory>(
                    &library,
                    b"hipImportExternalMemory\0",
                )?;
                let external_memory_get_mapped_buffer =
                    load_required::<HipExternalMemoryGetMappedBuffer>(
                        &library,
                        b"hipExternalMemoryGetMappedBuffer\0",
                    )?;
                let destroy_external_memory = load_required::<HipDestroyExternalMemory>(
                    &library,
                    b"hipDestroyExternalMemory\0",
                )?;
                let memset = load_required::<HipMemset>(&library, b"hipMemset\0")?;
                let device_synchronize =
                    load_required::<HipDeviceSynchronize>(&library, b"hipDeviceSynchronize\0")?;
                let get_error_string = library
                    .get::<HipGetErrorString>(b"hipGetErrorString\0")
                    .ok()
                    .map(|symbol| *symbol);

                let runtime = Self {
                    _library: library,

                    get_device_count,
                    device_get_name,
                    set_device,
                    import_external_memory,
                    external_memory_get_mapped_buffer,
                    destroy_external_memory,
                    memset,
                    device_synchronize,
                    get_error_string,
                };
                runtime.check(init(0), "hipInit")?;
                Ok(runtime)
            }
        }

        fn check(&self, code: HipError, operation: &str) -> Result<(), BackendError> {
            if code == HIP_SUCCESS {
                return Ok(());
            }

            let detail = self
                .get_error_string
                .and_then(|function| {
                    // SAFETY: HIP owns the returned static string. We only read it
                    // when the runtime returned a non-null pointer.
                    let pointer = unsafe { function(code) };
                    if pointer.is_null() {
                        None
                    } else {
                        Some(
                            unsafe { CStr::from_ptr(pointer) }
                                .to_string_lossy()
                                .into_owned(),
                        )
                    }
                })
                .unwrap_or_else(|| "unknown HIP error".to_owned());

            Err(BackendError::Internal(format!(
                "{operation} failed with HIP error {code}: {detail}"
            )))
        }

        fn devices(&self) -> Result<Vec<String>, BackendError> {
            let mut count = 0_i32;
            // SAFETY: pointer targets a valid initialized i32.
            self.check(
                unsafe { (self.get_device_count)(&mut count) },
                "hipGetDeviceCount",
            )?;
            if count < 0 {
                return Err(BackendError::Probe(
                    "HIP returned a negative device count".to_owned(),
                ));
            }

            let mut names = Vec::with_capacity(count as usize);
            for index in 0..count {
                let mut buffer = [0_i8; 256];
                // SAFETY: buffer is writable for the advertised length.
                self.check(
                    unsafe {
                        (self.device_get_name)(buffer.as_mut_ptr(), buffer.len() as i32, index)
                    },
                    "hipDeviceGetName",
                )?;
                // SAFETY: HIP documents hipDeviceGetName as NUL terminating the
                // supplied buffer when it succeeds.
                let name = unsafe { CStr::from_ptr(buffer.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                names.push(name);
            }
            Ok(names)
        }

        fn select_device(&self, vulkan_name: &str) -> Result<(i32, String), BackendError> {
            let devices = self.devices()?;
            if devices.is_empty() {
                return Err(BackendError::Unavailable(
                    "HIP runtime initialized but exposed no devices".to_owned(),
                ));
            }

            let target = normalize_device_name(vulkan_name);
            let selected = devices
                .iter()
                .enumerate()
                .find(|(_, name)| normalize_device_name(name) == target)
                .or_else(|| {
                    devices.iter().enumerate().find(|(_, name)| {
                        let candidate = normalize_device_name(name);
                        candidate.contains(&target) || target.contains(&candidate)
                    })
                })
                .or_else(|| (devices.len() == 1).then(|| (0, &devices[0])))
                .ok_or_else(|| {
                    BackendError::Unavailable(format!(
                        "could not correlate Vulkan device '{vulkan_name}' with HIP devices: {}",
                        devices.join(", ")
                    ))
                })?;

            let index = selected.0 as i32;
            // SAFETY: index came from the current HIP device inventory.
            self.check(unsafe { (self.set_device)(index) }, "hipSetDevice")?;
            Ok((index, selected.1.clone()))
        }

        fn import_memory(
            &self,
            handle: vk::HANDLE,
            allocation_size: u64,
        ) -> Result<HipExternalMemory, BackendError> {
            let descriptor = HipExternalMemoryHandleDesc {
                type_: HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT,
                handle: HipExternalMemoryHandleValue {
                    win32: HipWin32Handle {
                        handle: handle as usize as *mut c_void,
                        name: ptr::null(),
                    },
                },
                size: allocation_size,
                flags: 0,
                reserved: [0; 16],
            };
            let mut external_memory = ptr::null_mut();
            // SAFETY: descriptor mirrors hipExternalMemoryHandleDesc from the
            // public HIP ABI and the KMT handle remains backed by live Vulkan
            // DeviceMemory for the complete imported-memory lifetime.
            self.check(
                unsafe { (self.import_external_memory)(&mut external_memory, &descriptor) },
                "hipImportExternalMemory",
            )?;
            if external_memory.is_null() {
                return Err(BackendError::Internal(
                    "hipImportExternalMemory succeeded but returned a null handle".to_owned(),
                ));
            }
            Ok(external_memory)
        }

        fn map_external_memory(
            &self,
            external_memory: HipExternalMemory,
            bytes: u64,
        ) -> Result<*mut c_void, BackendError> {
            let descriptor = HipExternalMemoryBufferDesc {
                offset: 0,
                size: bytes,
                flags: 0,
                reserved: [0; 16],
            };
            let mut pointer = ptr::null_mut();
            // SAFETY: external_memory is live and descriptor describes a range
            // contained by the imported allocation.
            self.check(
                unsafe {
                    (self.external_memory_get_mapped_buffer)(
                        &mut pointer,
                        external_memory,
                        &descriptor,
                    )
                },
                "hipExternalMemoryGetMappedBuffer",
            )?;
            if pointer.is_null() {
                return Err(BackendError::Internal(
                    "HIP mapped external memory to a null device pointer".to_owned(),
                ));
            }
            Ok(pointer)
        }
    }

    struct HipExternalMemoryGuard<'a> {
        runtime: &'a HipRuntime,
        handle: HipExternalMemory,
    }

    impl Drop for HipExternalMemoryGuard<'_> {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                // SAFETY: guard owns exactly one imported HIP external-memory
                // handle and destroys it once before the Vulkan allocation dies.
                let _ = unsafe { (self.runtime.destroy_external_memory)(self.handle) };
            }
        }
    }

    struct VulkanContext {
        _entry: Entry,
        instance: Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        external_memory_win32: khr::external_memory_win32::Device,
        device_name: String,
    }

    impl VulkanContext {
        fn create() -> Result<Self, BackendError> {
            // SAFETY: loads the system Vulkan loader.
            let entry = unsafe { Entry::load() }
                .map_err(|error| BackendError::Unavailable(format!("Vulkan loader: {error}")))?;
            let app_name = CString::new("vulkan-rocm-bridge-smoke")
                .map_err(|error| BackendError::Internal(error.to_string()))?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(1)
                .engine_name(&app_name)
                .engine_version(1)
                .api_version(vk::API_VERSION_1_1);
            let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
            // SAFETY: all referenced create data lives through the call.
            let instance = unsafe { entry.create_instance(&instance_info, None) }
                .map_err(|error| BackendError::Probe(format!("vkCreateInstance: {error:?}")))?;

            let selection = select_vulkan_device(&instance);
            let (physical_device, queue_family_index, device_name) = match selection {
                Ok(selection) => selection,
                Err(error) => {
                    // SAFETY: no device children exist yet.
                    unsafe { instance.destroy_instance(None) };
                    return Err(error);
                }
            };

            let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;
            let usage = shared_buffer_usage();
            let external_info = vk::PhysicalDeviceExternalBufferInfo::default()
                .flags(vk::BufferCreateFlags::empty())
                .usage(usage)
                .handle_type(handle_type);
            let mut external_properties = vk::ExternalBufferProperties::default();
            // SAFETY: physical_device belongs to this instance and output is valid.
            unsafe {
                instance.get_physical_device_external_buffer_properties(
                    physical_device,
                    &external_info,
                    &mut external_properties,
                )
            };
            let features = external_properties
                .external_memory_properties
                .external_memory_features;
            if !features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE) {
                // SAFETY: no device children exist yet.
                unsafe { instance.destroy_instance(None) };
                return Err(BackendError::Unsupported(format!(
                    "Vulkan device '{device_name}' cannot export OPAQUE_WIN32_KMT buffer memory"
                )));
            }

            let priorities = [1.0_f32];
            let queue_infos = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let extension_names = [khr::external_memory_win32::NAME.as_ptr()];
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&extension_names);
            // SAFETY: physical_device and create arrays are valid for this call.
            let device =
                match unsafe { instance.create_device(physical_device, &device_info, None) } {
                    Ok(device) => device,
                    Err(error) => {
                        // SAFETY: no device children exist.
                        unsafe { instance.destroy_instance(None) };
                        return Err(BackendError::Probe(format!(
                            "vkCreateDevice for '{device_name}': {error:?}"
                        )));
                    }
                };

            // SAFETY: queue family was selected from this physical device and one
            // queue was requested from that family.
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            // SAFETY: device is valid and queue family exists.
            let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
                Ok(pool) => pool,
                Err(error) => {
                    // SAFETY: device has no surviving children at this point.
                    unsafe {
                        device.destroy_device(None);
                        instance.destroy_instance(None);
                    }
                    return Err(BackendError::Probe(format!(
                        "vkCreateCommandPool: {error:?}"
                    )));
                }
            };
            let external_memory_win32 = khr::external_memory_win32::Device::new(&instance, &device);

            Ok(Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue_family_index,
                queue,
                command_pool,
                external_memory_win32,
                device_name,
            })
        }

        fn create_shared_buffer(&self, bytes: u64) -> Result<VulkanBuffer, BackendError> {
            let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;
            let mut external =
                vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(shared_buffer_usage())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external);
            // SAFETY: create data is valid for the duration of the call.
            let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }
                .map_err(|error| BackendError::Internal(format!("vkCreateBuffer: {error:?}")))?;

            // SAFETY: buffer belongs to this device.
            let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let memory_properties = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            let memory_type_index = find_memory_type(
                &memory_properties,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                // SAFETY: buffer is live and owned by this device.
                unsafe { self.device.destroy_buffer(buffer, None) };
                BackendError::Unsupported(
                    "no device-local memory type satisfies the exportable buffer".to_owned(),
                )
            })?;

            let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index)
                .push_next(&mut dedicated)
                .push_next(&mut export);
            // SAFETY: memory type and allocation size come from this buffer's requirements.
            let memory = match unsafe { self.device.allocate_memory(&allocate_info, None) } {
                Ok(memory) => memory,
                Err(error) => {
                    // SAFETY: buffer is live and unbound.
                    unsafe { self.device.destroy_buffer(buffer, None) };
                    return Err(BackendError::Internal(format!(
                        "vkAllocateMemory(exportable): {error:?}"
                    )));
                }
            };
            // SAFETY: allocation satisfies this buffer's requirements and offset is aligned.
            if let Err(error) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
                // SAFETY: neither resource is in use.
                unsafe {
                    self.device.free_memory(memory, None);
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(BackendError::Internal(format!(
                    "vkBindBufferMemory(exportable): {error:?}"
                )));
            }

            Ok(VulkanBuffer {
                device: self.device.clone(),
                buffer,
                memory,
                logical_size: bytes,
                allocation_size: requirements.size,
            })
        }

        fn create_staging_buffer(&self, bytes: u64) -> Result<VulkanBuffer, BackendError> {
            let info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            // SAFETY: create data is valid.
            let buffer = unsafe { self.device.create_buffer(&info, None) }.map_err(|error| {
                BackendError::Internal(format!("staging vkCreateBuffer: {error:?}"))
            })?;
            // SAFETY: buffer belongs to this device.
            let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let memory_properties = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            let required =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let memory_type_index =
                find_memory_type(&memory_properties, requirements.memory_type_bits, required)
                    .ok_or_else(|| {
                        // SAFETY: buffer is live and unbound.
                        unsafe { self.device.destroy_buffer(buffer, None) };
                        BackendError::Unsupported(
                            "no HOST_VISIBLE|HOST_COHERENT staging memory type is available"
                                .to_owned(),
                        )
                    })?;
            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index);
            // SAFETY: allocation parameters come from the buffer requirements.
            let memory = match unsafe { self.device.allocate_memory(&allocate_info, None) } {
                Ok(memory) => memory,
                Err(error) => {
                    // SAFETY: buffer is live and unbound.
                    unsafe { self.device.destroy_buffer(buffer, None) };
                    return Err(BackendError::Internal(format!(
                        "staging vkAllocateMemory: {error:?}"
                    )));
                }
            };
            // SAFETY: allocation satisfies buffer requirements.
            if let Err(error) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
                // SAFETY: resources are not in use.
                unsafe {
                    self.device.free_memory(memory, None);
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(BackendError::Internal(format!(
                    "staging vkBindBufferMemory: {error:?}"
                )));
            }

            Ok(VulkanBuffer {
                device: self.device.clone(),
                buffer,
                memory,
                logical_size: bytes,
                allocation_size: requirements.size,
            })
        }

        fn export_kmt_handle(&self, shared: &VulkanBuffer) -> Result<vk::HANDLE, BackendError> {
            let info = vk::MemoryGetWin32HandleInfoKHR::default()
                .memory(shared.memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT);
            // SAFETY: shared.memory was allocated with the matching export handle type.
            unsafe { self.external_memory_win32.get_memory_win32_handle(&info) }.map_err(|error| {
                BackendError::Internal(format!("vkGetMemoryWin32HandleKHR: {error:?}"))
            })
        }

        fn release_to_external(&self, shared: &VulkanBuffer) -> Result<(), BackendError> {
            self.submit_one_time(|command_buffer| {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .buffer(shared.buffer)
                    .offset(0)
                    .size(shared.logical_size);
                // SAFETY: command buffer is recording, shared buffer is valid, and
                // this is a release ownership transfer to the external queue family.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    )
                };
            })
        }

        fn acquire_and_copy_to_staging(
            &self,
            shared: &VulkanBuffer,
            staging: &VulkanBuffer,
        ) -> Result<(), BackendError> {
            self.submit_one_time(|command_buffer| {
                let acquire = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .dst_queue_family_index(self.queue_family_index)
                    .buffer(shared.buffer)
                    .offset(0)
                    .size(shared.logical_size);
                // SAFETY: command buffer is recording and this acquires external
                // ownership before the transfer read.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[acquire],
                        &[],
                    )
                };

                let region = vk::BufferCopy::default().size(shared.logical_size);
                // SAFETY: both buffers are live, correctly bound, and sized for the copy.
                unsafe {
                    self.device.cmd_copy_buffer(
                        command_buffer,
                        shared.buffer,
                        staging.buffer,
                        &[region],
                    )
                };

                let host_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .buffer(staging.buffer)
                    .offset(0)
                    .size(staging.logical_size);
                // SAFETY: makes the transfer write visible to subsequent host reads.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[host_barrier],
                        &[],
                    )
                };
            })
        }

        fn verify_staging(&self, staging: &VulkanBuffer, pattern: u8) -> Result<u64, BackendError> {
            // SAFETY: staging memory is HOST_VISIBLE and HOST_COHERENT and no GPU
            // submission remains in flight because submit_one_time waits for queue idle.
            let pointer = unsafe {
                self.device.map_memory(
                    staging.memory,
                    0,
                    staging.logical_size,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|error| BackendError::Internal(format!("vkMapMemory(staging): {error:?}")))?;

            // SAFETY: mapped range is at least logical_size bytes and remains mapped
            // until after the slice is fully inspected.
            let bytes = unsafe {
                slice::from_raw_parts(pointer.cast::<u8>(), staging.logical_size as usize)
            };
            let mismatch = bytes
                .iter()
                .position(|value| *value != pattern)
                .map(|index| (index, bytes[index]));
            // SAFETY: pointer belongs to this memory mapping and is unmapped once.
            unsafe { self.device.unmap_memory(staging.memory) };

            if let Some((index, actual)) = mismatch {
                return Err(BackendError::Internal(format!(
                    "zero-copy verification failed at byte {index}: expected 0x{pattern:02x}, got 0x{actual:02x}"
                )));
            }
            Ok(staging.logical_size)
        }

        fn submit_one_time<F>(&self, record: F) -> Result<(), BackendError>
        where
            F: FnOnce(vk::CommandBuffer),
        {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: command pool is valid and belongs to this device.
            let command_buffers = unsafe { self.device.allocate_command_buffers(&allocate_info) }
                .map_err(|error| {
                BackendError::Internal(format!("vkAllocateCommandBuffers: {error:?}"))
            })?;
            let command_buffer = command_buffers[0];
            let result = (|| {
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                // SAFETY: command buffer is freshly allocated and not recording.
                unsafe { self.device.begin_command_buffer(command_buffer, &begin) }.map_err(
                    |error| BackendError::Internal(format!("vkBeginCommandBuffer: {error:?}")),
                )?;
                record(command_buffer);
                // SAFETY: recorded commands are complete.
                unsafe { self.device.end_command_buffer(command_buffer) }.map_err(|error| {
                    BackendError::Internal(format!("vkEndCommandBuffer: {error:?}"))
                })?;
                let command_buffer_list = [command_buffer];
                let submit = [vk::SubmitInfo::default().command_buffers(&command_buffer_list)];
                // SAFETY: queue and command buffer belong to this device. No fence
                // is needed because we synchronously wait for the queue immediately.
                unsafe {
                    self.device
                        .queue_submit(self.queue, &submit, vk::Fence::null())
                }
                .map_err(|error| BackendError::Internal(format!("vkQueueSubmit: {error:?}")))?;
                // SAFETY: queue is valid; wait establishes host synchronization.
                unsafe { self.device.queue_wait_idle(self.queue) }.map_err(|error| {
                    BackendError::Internal(format!("vkQueueWaitIdle: {error:?}"))
                })?;
                Ok(())
            })();

            // SAFETY: queue is idle on the success path; Vulkan also permits freeing
            // a command buffer after failed recording before it was submitted.
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &command_buffers)
            };
            result
        }
    }

    impl Drop for VulkanContext {
        fn drop(&mut self) {
            // SAFETY: callers declare buffers after the context, so their Drop runs
            // first. We still wait idle defensively before tearing down the pool/device.
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_command_pool(self.command_pool, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    struct VulkanBuffer {
        device: Device,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        logical_size: u64,
        allocation_size: u64,
    }

    impl Drop for VulkanBuffer {
        fn drop(&mut self) {
            // SAFETY: owning context remains alive until after these buffers drop.
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    pub(super) fn run(bytes: u64, pattern: u8) -> Result<ZeroCopySmokeReport, BackendError> {
        if bytes == 0 {
            return Err(BackendError::Internal(
                "zero-copy smoke size must be greater than zero".to_owned(),
            ));
        }
        let max_bytes = env::var("VRB_MAX_SMOKE_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_SMOKE_BYTES);
        if bytes > max_bytes {
            return Err(BackendError::Unsupported(format!(
                "requested {bytes} bytes exceeds VRB_MAX_SMOKE_BYTES={max_bytes}"
            )));
        }
        let usize_bytes = usize::try_from(bytes)
            .map_err(|_| BackendError::Unsupported("smoke size does not fit usize".to_owned()))?;

        let vulkan = VulkanContext::create()?;
        let shared = vulkan.create_shared_buffer(bytes)?;
        let staging = vulkan.create_staging_buffer(bytes)?;
        let handle = vulkan.export_kmt_handle(&shared)?;

        let hip = HipRuntime::load()?;
        let (_hip_index, hip_device) = hip.select_device(&vulkan.device_name)?;

        vulkan.release_to_external(&shared)?;
        let external_memory = hip.import_memory(handle, shared.allocation_size)?;
        let external_memory_guard = HipExternalMemoryGuard {
            runtime: &hip,
            handle: external_memory,
        };
        let hip_pointer = hip.map_external_memory(external_memory_guard.handle, bytes)?;
        // SAFETY: mapped device pointer represents at least `bytes` bytes in the
        // imported Vulkan allocation; pattern is converted to hipMemset's int value.
        hip.check(
            unsafe { (hip.memset)(hip_pointer, i32::from(pattern), usize_bytes) },
            "hipMemset(shared Vulkan memory)",
        )?;
        // SAFETY: runtime is initialized and selected device is current.
        hip.check(
            unsafe { (hip.device_synchronize)() },
            "hipDeviceSynchronize",
        )?;
        drop(external_memory_guard);

        vulkan.acquire_and_copy_to_staging(&shared, &staging)?;
        let verified_bytes = vulkan.verify_staging(&staging, pattern)?;

        Ok(ZeroCopySmokeReport {
            vulkan_device: vulkan.device_name.clone(),
            hip_device,
            bytes,
            pattern,
            verified_bytes,
            external_memory_handle:
                "VK_OPAQUE_WIN32_KMT / hipExternalMemoryHandleTypeOpaqueWin32Kmt",
            synchronization:
                "Vulkan EXTERNAL queue-family ownership + host queue/HIP synchronization",
        })
    }

    fn select_vulkan_device(
        instance: &Instance,
    ) -> Result<(vk::PhysicalDevice, u32, String), BackendError> {
        // SAFETY: instance is valid.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|error| BackendError::Probe(format!("enumerate Vulkan devices: {error:?}")))?;

        let mut candidates = Vec::new();
        for physical_device in physical_devices {
            // SAFETY: handle came from this instance.
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            // SAFETY: Vulkan guarantees NUL termination.
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: handle came from this instance.
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let Some((queue_family_index, _)) =
                queue_families.iter().enumerate().find(|(_, family)| {
                    family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
            else {
                continue;
            };
            // SAFETY: handle came from this instance.
            let extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }
                    .map_err(|error| {
                        BackendError::Probe(format!("enumerate Vulkan extensions: {error:?}"))
                    })?;
            let has_external_memory_win32 = extensions.iter().any(|extension| {
                // SAFETY: Vulkan guarantees NUL termination.
                (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
                    == khr::external_memory_win32::NAME
            });
            if !has_external_memory_win32 {
                continue;
            }

            let rank = (
                if properties.vendor_id == AMD_PCI_VENDOR_ID {
                    0_u8
                } else {
                    1
                },
                if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    0_u8
                } else {
                    1
                },
                properties.device_id,
            );
            candidates.push((rank, physical_device, queue_family_index as u32, name));
        }

        candidates.sort_by_key(|candidate| candidate.0);
        candidates
            .into_iter()
            .next()
            .map(|(_, device, queue, name)| (device, queue, name))
            .ok_or_else(|| {
                BackendError::Unavailable(
                    "no compute-capable Vulkan device with VK_KHR_external_memory_win32 was found"
                        .to_owned(),
                )
            })
    }

    fn shared_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
    }

    fn find_memory_type(
        properties: &vk::PhysicalDeviceMemoryProperties,
        memory_type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..properties.memory_type_count).find(|index| {
            let index_value = *index;
            let supported = memory_type_bits & (1_u32 << index_value) != 0;
            let flags = properties.memory_types[index_value as usize].property_flags;
            supported && flags.contains(required)
        })
    }

    fn normalize_device_name(value: &str) -> String {
        value
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    }

    fn hip_library_candidates() -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        for variable in ["HIP_PATH", "ROCM_PATH"] {
            if let Some(root) = env::var_os(variable) {
                candidates.push(PathBuf::from(&root).join("bin").join("amdhip64.dll"));
            }
        }
        candidates.push(PathBuf::from("amdhip64.dll"));
        candidates
    }

    fn load_hip_library() -> Result<(Library, String), BackendError> {
        let mut failures = Vec::new();
        for candidate in hip_library_candidates() {
            // SAFETY: candidates are restricted to the official HIP runtime name
            // and administrator-selected HIP_PATH/ROCM_PATH roots.
            match unsafe { Library::new(&candidate) } {
                Ok(library) => return Ok((library, candidate.display().to_string())),
                Err(error) => failures.push(format!("{}: {error}", candidate.display())),
            }
        }
        Err(BackendError::Unavailable(format!(
            "HIP runtime library was not loadable for zero-copy smoke test ({})",
            failures.join("; ")
        )))
    }

    unsafe fn load_required<T: Copy>(library: &Library, name: &[u8]) -> Result<T, BackendError> {
        // SAFETY: caller specifies the exact function-pointer type for the public
        // HIP C symbol and retains the library while the copied pointer is used.
        unsafe { library.get::<T>(name) }
            .map(|symbol| *symbol)
            .map_err(|error| {
                BackendError::Unavailable(format!(
                    "required HIP symbol '{}' is missing: {error}",
                    String::from_utf8_lossy(name).trim_end_matches('\0')
                ))
            })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn hip_external_memory_abi_has_expected_win64_size() {
            // hipExternalMemoryHandleDesc: 4-byte enum + alignment + 16-byte union
            // + u64 size + u32 flags + 16*u32 reserved = 104 bytes on Win64.
            assert_eq!(std::mem::size_of::<HipExternalMemoryHandleDesc>(), 104);
            assert_eq!(std::mem::size_of::<HipExternalMemoryBufferDesc>(), 80);
        }

        #[test]
        fn device_name_normalization_is_stable() {
            assert_eq!(
                normalize_device_name("AMD Radeon RX 6800 XT"),
                "amdradeonrx6800xt"
            );
        }
    }
}
