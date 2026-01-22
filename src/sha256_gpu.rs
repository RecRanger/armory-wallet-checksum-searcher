// https://sotrh.github.io/learn-wgpu/compute/introduction/

use enum_map::{Enum, EnumMap, enum_map};
use once_cell::sync::OnceCell;
use std::sync::{Arc, Mutex};

/// Calculate the length of the padded message.
///
/// Args:
/// - `len_bytes`: Length of the input message (before any padding).
///
/// Returns the padded length of the message.
const fn calc_padded_message_length_bytes(len_bytes: u32) -> u32 {
    let len_bits = len_bytes * 8;

    // Number of bits to add so that:
    // len + 1 + k + 64 ≡ 0 (mod 512)
    let k = (512 - ((len_bits + 1 + 64) % 512)) % 512;
    let total_bits = len_bits + 1 + k + 64;

    let total_bytes_with_padding = total_bits / 8;

    total_bytes_with_padding
}

/// Pad a message for SHA256 specifically (including appending its length).
///
/// Writes the padded message into `output_words`.
/// The `output_words` slice should be `padded_len_bytes / 4` elements long.
/// Supports non-zero-filled data.
fn pad_message_into<M: AsRef<[u8]>>(message_bytes: M, output_words: &mut [u32]) {
    let output_bytes = bytemuck::cast_slice_mut::<u32, u8>(output_words);

    let message_bytes_len: usize = message_bytes.as_ref().len();

    // Copy message.
    output_bytes[..message_bytes_len].copy_from_slice(message_bytes.as_ref());

    // Append the 0x80 byte.
    output_bytes[message_bytes_len] = 0x80;

    // Zero only the required padding bytes (between message and the final length u64).
    let len_pos = output_bytes.len() - 8;
    output_bytes[(message_bytes_len + 1)..len_pos].fill(0);

    // Append message length in bits (big endian, 64 bit value, u64).
    let bit_len = (message_bytes_len as u64) * 8;
    output_bytes[len_pos..].copy_from_slice(&bit_len.to_be_bytes());
}

fn calc_num_workgroups(device: &wgpu::Device, num_messages: u32) -> u32 {
    let max_wg = device.limits().max_compute_workgroup_size_x;
    let max_groups = device.limits().max_compute_workgroups_per_dimension;

    let num = ((num_messages as u32) + max_wg - 1) / max_wg;
    if num > max_groups {
        panic!("Input array too large. Max size is {}", max_groups * max_wg);
    }
    num
}

struct GpuBuffers {
    // Overall data path message_upload_buffer -> message_buffer -> result_buffer -> readback_buffer.
    message_upload_buffer: wgpu::Buffer,
    message_upload_capacity_bytes: u64,

    message_buffer: wgpu::Buffer,
    message_buffer_capacity_bytes: u64,

    result_buffer: wgpu::Buffer,
    result_buffer_capacity_bytes: u64,

    readback_buffer: wgpu::Buffer,
    readback_buffer_capacity_bytes: u64,

    num_messages_buffer: wgpu::Buffer,
    message_sizes_buffer: wgpu::Buffer,

    bind_group: wgpu::BindGroup,
}

fn ensure_buffer_capacity(
    device: &wgpu::Device,
    existing: Option<wgpu::Buffer>,
    existing_capacity: u64,
    required_capacity: u64,
    usage: wgpu::BufferUsages,
    label: &str,
    mapped_at_creation: bool,
) -> (wgpu::Buffer, u64) {
    if let Some(buffer) = existing {
        if existing_capacity >= required_capacity {
            return (buffer, existing_capacity);
        }
    }

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: required_capacity,
        usage,
        mapped_at_creation,
    });

    (buffer, required_capacity)
}

#[derive(Clone, Copy, Enum)]
pub enum ComputePipelineVersion {
    Sha256,
    Sha256d,
}

struct GPU {
    device: wgpu::Device,
    queue: wgpu::Queue,

    compute_pipelines: EnumMap<ComputePipelineVersion, wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,

    buffers: Mutex<Option<GpuBuffers>>,

    scratch_upload_buffer: Mutex<Vec<u32>>,
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,

    version: ComputePipelineVersion,
) -> wgpu::ComputePipeline {
    // Inject configuration values into sha256-gpu.
    let pipeline_constants: &[(&str, f64)] = &[
        (
            &"CONFIG_WORKGROUP_SIZE",
            device.limits().max_compute_workgroup_size_x as f64,
        ),
        (
            &"CONFIG_ENABLE_SHA256D",
            match version {
                ComputePipelineVersion::Sha256 => 0.0f64,
                ComputePipelineVersion::Sha256d => 1.0f64,
            },
        ),
    ];

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: match version {
            ComputePipelineVersion::Sha256 => Some("SHA256 Compute Pipeline"),
            ComputePipelineVersion::Sha256d => Some("SHA256d Compute Pipeline"),
        },
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: Some("sha256_or_sha256d"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: pipeline_constants,
            ..Default::default()
        },
        cache: Default::default(),
    });
    compute_pipeline
}

impl GPU {
    async fn init() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        // Alternative default option:
        // let adapter = instance.request_adapter(&Default::default()).await.unwrap();

        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        let shader = device.create_shader_module(wgpu::include_wgsl!("sha256-gpu.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SHA256 Bind Group Layout"),
            entries: &[
                // messages
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // num_messages
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // message_sizes
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // hashes
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SHA256x Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let compute_pipelines = enum_map! {
            ComputePipelineVersion::Sha256 => create_compute_pipeline(
                &device, &shader, &pipeline_layout, ComputePipelineVersion::Sha256
            ),
            ComputePipelineVersion::Sha256d => create_compute_pipeline(
                &device, &shader, &pipeline_layout, ComputePipelineVersion::Sha256d
            ),
        };

        Ok(Self {
            device,
            queue,
            compute_pipelines,
            bind_group_layout,
            buffers: Mutex::new(None),
            scratch_upload_buffer: Mutex::new(Vec::new()),
        })
    }

    fn pipeline_for(&self, algo: ComputePipelineVersion) -> &wgpu::ComputePipeline {
        &self.compute_pipelines[algo]
    }
}

static GPU_INSTANCE: OnceCell<GPU> = OnceCell::new();
fn get_gpu() -> anyhow::Result<&'static GPU> {
    GPU_INSTANCE.get_or_try_init(|| pollster::block_on(GPU::init()))
}

pub async fn sha256_gpu<const N: usize, M: AsRef<[u8]>>(
    messages: &[M],
    compute_pipeline_version: ComputePipelineVersion,
) -> anyhow::Result<Vec<[u8; N]>>
where
    [u8; N]: Sized,
{
    // static_assertions::const_assert!(N <= 32); // TODO: Enable when in stable Rust.

    let gpu = get_gpu()?;

    let num_messages: u32 = messages.len() as u32;
    let num_workgroups = calc_num_workgroups(&gpu.device, num_messages);

    let message_length: u32 = messages[0].as_ref().len() as u32;
    let padded_message_length_bytes = calc_padded_message_length_bytes(message_length);
    let padded_message_length_words: usize = (padded_message_length_bytes as usize) / 4;

    // Pack messages into an allocated once-per-message size array.
    let target_message_array_total_length_words = messages.len() * padded_message_length_words;
    let mut scratch_upload_buffer_guard: std::sync::MutexGuard<'_, Vec<u32>> =
        match gpu.scratch_upload_buffer.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
    if scratch_upload_buffer_guard.len() != target_message_array_total_length_words {
        scratch_upload_buffer_guard.resize(target_message_array_total_length_words, 0);
    }
    for (message_index, message_bytes) in messages.iter().enumerate() {
        // Validate that all messages are the same length! Critical requirement.
        if message_bytes.as_ref().len() != message_length as usize {
            panic!(
                "All messages must have the same length. Message #{}'s length is {} but expected {}.",
                message_index,
                message_bytes.as_ref().len(),
                message_length
            );
        }

        // Pad. Only pass in the slice to write to.
        let start = message_index * padded_message_length_words;
        let end = start + padded_message_length_words;
        pad_message_into(message_bytes, &mut scratch_upload_buffer_guard[start..end]);
    }

    // let mut gpu_buffers_guard = gpu.buffers.lock().unwrap();
    let mut gpu_buffers_guard = match gpu.buffers.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    let padded_messages_total_bytes = (messages.len() * padded_message_length_words * 4) as u64;
    let result_size_bytes = (32 * messages.len()) as u64;

    let buffers = gpu_buffers_guard.get_or_insert_with(|| {
        let message_upload_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("message_upload"),
            size: padded_messages_total_bytes,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false, // Important.
        });

        let message_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("messages"),
            size: padded_messages_total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashes"),
            size: result_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: result_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let num_messages_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("num_messages"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let message_sizes_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("message_sizes"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gpu.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: message_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: num_messages_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: message_sizes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
            label: Some("sha256-bind-group"),
        });

        GpuBuffers {
            message_upload_buffer,
            message_upload_capacity_bytes: padded_messages_total_bytes,
            message_buffer,
            message_buffer_capacity_bytes: padded_messages_total_bytes,
            result_buffer,
            result_buffer_capacity_bytes: result_size_bytes,
            readback_buffer,
            readback_buffer_capacity_bytes: result_size_bytes,
            num_messages_buffer,
            message_sizes_buffer,
            bind_group,
        }
    });

    let (message_upload_buffer, message_upload_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.message_upload_buffer.clone()),
        buffers.message_upload_capacity_bytes,
        padded_messages_total_bytes,
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        "message_upload",
        false, // Important.
    );

    let (message_buffer, message_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.message_buffer.clone()),
        buffers.message_buffer_capacity_bytes,
        padded_messages_total_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "messages",
        false,
    );

    let (result_buffer, result_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.result_buffer.clone()),
        buffers.result_buffer_capacity_bytes,
        result_size_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "hashes",
        false,
    );

    let (readback_buffer, readback_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.readback_buffer.clone()),
        buffers.readback_buffer_capacity_bytes,
        result_size_bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "readback",
        false,
    );

    if message_upload_capacity != buffers.message_upload_capacity_bytes
        || message_capacity != buffers.message_buffer_capacity_bytes
        || result_capacity != buffers.result_buffer_capacity_bytes
        || readback_capacity != buffers.readback_buffer_capacity_bytes
    {
        println!("Recreating bind group due to buffer capacity change");

        buffers.bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gpu.bind_group_layout,
            entries: &[
                // Note: We exclude the message_upload_buffer and the readback_buffer here.
                // Those buffers are CPU-size only, and only accessed by copying.
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: message_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.num_messages_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.message_sizes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        // Update in buffers
        buffers.message_upload_buffer = message_upload_buffer;
        buffers.message_upload_capacity_bytes = message_upload_capacity;
        buffers.message_buffer = message_buffer;
        buffers.message_buffer_capacity_bytes = message_capacity;
        buffers.result_buffer = result_buffer;
        buffers.result_buffer_capacity_bytes = result_capacity;
        buffers.readback_buffer = readback_buffer;
        buffers.readback_buffer_capacity_bytes = readback_capacity;
    }

    // gpu.queue.write_buffer(
    //     &buffers.message_buffer,
    //     0,
    //     bytemuck::cast_slice(&message_array),
    // );

    let upload_buffer_slice = buffers
        .message_upload_buffer
        .slice(0..padded_messages_total_bytes);

    let upload_map_done = Arc::new(Mutex::new(None));
    let upload_map_done_clone = upload_map_done.clone();
    upload_buffer_slice.map_async(wgpu::MapMode::Write, move |result| {
        *upload_map_done_clone.lock().unwrap() = Some(result);
    });

    // VERY IMPORTANT: Drive the mapping to completion.
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

    // Check mapping result.
    match upload_map_done.lock().unwrap().take().unwrap() {
        Ok(()) => {}
        Err(e) => return Err(e.into()),
    }

    {
        let mut mapped_range = upload_buffer_slice.get_mapped_range_mut();

        let dst_u32: &mut [u32] = bytemuck::cast_slice_mut(&mut mapped_range[..]);

        dst_u32[..target_message_array_total_length_words]
            .copy_from_slice(&scratch_upload_buffer_guard);
    }
    buffers.message_upload_buffer.unmap();

    gpu.queue.write_buffer(
        &buffers.num_messages_buffer,
        0,
        bytemuck::cast_slice(&[num_messages]),
    );

    // TODO: Maybe pass these in as two separate inputs.
    // `message_sizes[0]` is the original message length in bytes.
    // `message_sizes[1]` is the padded message length in bytes.
    gpu.queue.write_buffer(
        &buffers.message_sizes_buffer,
        0,
        bytemuck::cast_slice(&[message_length, padded_message_length_bytes]),
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    encoder.copy_buffer_to_buffer(
        &buffers.message_upload_buffer,
        0,
        &buffers.message_buffer,
        0,
        padded_messages_total_bytes,
    );

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&gpu.pipeline_for(compute_pipeline_version));
        pass.set_bind_group(0, &buffers.bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &buffers.result_buffer,
        0,
        &buffers.readback_buffer,
        0,
        result_size_bytes,
    );
    gpu.queue.submit(Some(encoder.finish()));

    // Read back from the GPU.
    // Must include upper bound here in case the readback_buffer is oversized.
    // Is a real case.
    let buffer_slice = buffers.readback_buffer.slice(0..result_size_bytes);

    let result_map_done = Arc::new(Mutex::new(None));
    let result_map_done_clone = result_map_done.clone();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        *result_map_done_clone.lock().unwrap() = Some(result);
    });

    // VERY IMPORTANT: Drive the mapping to completion.
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

    // Check mapping result.
    match result_map_done.lock().unwrap().take().unwrap() {
        Ok(()) => {}
        Err(e) => return Err(e.into()),
    }

    let mapped = buffer_slice.get_mapped_range();

    // Reshape from long output array into separate hash values for each message.
    const FULL_HASH_SIZE: usize = 32;
    let valid_bytes = &mapped[..result_size_bytes as usize];
    let hashes: Vec<[u8; N]> = {
        if N == 32 {
            bytemuck::cast_slice::<u8, [u8; N]>(valid_bytes).to_vec()
        } else if N > 32 {
            panic!("Hash size greater than 32 is not supported");
        } else {
            valid_bytes
                .chunks_exact(FULL_HASH_SIZE)
                .map(|full_hash_bytes| {
                    let mut truncated_hash = [0u8; N];
                    truncated_hash.copy_from_slice(&full_hash_bytes[..N]);
                    truncated_hash
                })
                .collect()
        }
    };

    debug_assert_eq!(hashes.len(), messages.len());

    drop(mapped);
    buffers.readback_buffer.unmap();

    Ok(hashes)
}

/// Logic implementation of max allowed...
///
/// Exists as a separate function for testability.
fn calc_allowed_message_count_per_operation(
    each_message_length: usize,
    device_max_wg_size_x: u32,
    device_max_wg_per_dimension: u32,
    device_max_buffer_sizes: &[u64],
) -> usize {
    use std::cmp::min;

    // Based on logic/docs in `calc_num_workgroups()`.
    let max_from_workgroups = (device_max_wg_per_dimension * device_max_wg_size_x) as usize;

    // Based on errors about max buffer size.
    let each_message_length_padded = calc_padded_message_length_bytes(each_message_length as u32);
    let device_max_buffer_size: u64 = *device_max_buffer_sizes
        .iter()
        .min() // Pick the tighter option.
        .unwrap(); // Unwrap is safe because the array should never be empty.
    let max_from_buffer = (device_max_buffer_size / each_message_length_padded as u64) as usize;

    // Combine. Select the lower value for safety.
    let max_overall = min(max_from_workgroups, max_from_buffer);
    max_overall
}

/// Fetch the maximum allowable number of messages that can be passsed to `run_sha256_gpu()`.
pub fn max_allowed_message_count_per_operation(
    each_message_length: usize,
) -> anyhow::Result<usize> {
    let device = &get_gpu()?.device;

    let device_max_wg_size_x = device.limits().max_compute_workgroup_size_x;
    let device_max_wg_per_dimension = device.limits().max_compute_workgroups_per_dimension;
    let device_max_buffer_sizes: &[u64] = &[
        device.limits().max_buffer_size,
        // All the `max_*_buffer_binding_size`
        device.limits().max_storage_buffer_binding_size as u64,
    ];

    Ok(calc_allowed_message_count_per_operation(
        each_message_length,
        device_max_wg_size_x,
        device_max_wg_per_dimension,
        device_max_buffer_sizes,
    ))
}

#[allow(dead_code)]
pub fn run_sha256_gpu<const N: usize, M: AsRef<[u8]>>(
    messages: &[M],
) -> anyhow::Result<Vec<[u8; N]>> {
    pollster::block_on(sha256_gpu::<N, M>(messages, ComputePipelineVersion::Sha256))
}

#[allow(dead_code)]
pub fn run_sha256d_gpu<const N: usize, M: AsRef<[u8]>>(
    messages: &[M],
) -> anyhow::Result<Vec<[u8; N]>> {
    pollster::block_on(sha256_gpu::<N, M>(
        messages,
        ComputePipelineVersion::Sha256d,
    ))
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use crate::{
        search_with_cpu::{sha256, sha256d},
        test_general::get_test_config,
    };

    use super::*;

    /// Same-length messages with `len(input) % 4 == 0` (aligned).
    #[test]
    fn test_sha256_is_valid_aligned() {
        let input_1 = b"Hello, wgsl\n";
        let expect_1: [u8; 32] = [
            254, 234, 146, 74, 232, 68, 234, 160, 191, 118, 232, 179, 211, 60, 233, 49, 144, 98,
            156, 231, 56, 159, 25, 217, 66, 189, 1, 131, 11, 167, 119, 180,
        ];

        let input_2 = b"Hello world\n";
        let expect_2: [u8; 32] = [
            24, 148, 161, 156, 133, 186, 21, 58, 203, 247, 67, 172, 78, 67, 252, 0, 76, 137, 22, 4,
            178, 111, 140, 105, 225, 232, 62, 162, 175, 199, 196, 143,
        ];

        let hashes = run_sha256_gpu(&[input_1, input_2]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expect_1, expect_2]);

        // Test again to ensure repeated calls work right.
        let hashes_again = run_sha256_gpu(&[input_1, input_2]).unwrap();
        assert_eq!(hashes_again.len(), 2);
        assert_eq!(hashes_again, vec![expect_1, expect_2]);
    }

    #[test]
    fn test_sha256_4bytes_is_valid_aligned() {
        let input_1 = b"Hello, wgsl\n";
        let expect_1: [u8; 4] = [254, 234, 146, 74];

        let input_2 = b"Hello world\n";
        let expect_2: [u8; 4] = [24, 148, 161, 156];

        let hashes: Vec<[u8; 4]> = run_sha256_gpu(&[input_1, input_2]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expect_1, expect_2]);

        // Test again to ensure repeated calls work right.
        let hashes_again: Vec<[u8; 4]> = run_sha256_gpu(&[input_1, input_2]).unwrap();
        assert_eq!(hashes_again.len(), 2);
        assert_eq!(hashes_again, vec![expect_1, expect_2]);
    }

    #[test]
    fn test_sha256_12_bytes_is_valid_aligned() {
        let input_1 = b"Hello, wgsl\n";
        let expect_1: [u8; 12] = [254, 234, 146, 74, 232, 68, 234, 160, 191, 118, 232, 179];

        let input_2 = b"Hello world\n";
        let expect_2: [u8; 12] = [24, 148, 161, 156, 133, 186, 21, 58, 203, 247, 67, 172];

        let hashes: Vec<[u8; 12]> = run_sha256_gpu(&[input_1, input_2]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expect_1, expect_2]);
    }

    /// Same-length messages with `len(input) % 4 != 0` (unaligned).
    #[test]
    fn test_sha256_is_valid_unaligned() {
        let input_1 = b"Hello world.\n";
        let expect_1: [u8; 32] = [
            // 6472bf692aaf270d5f9dc40c5ecab8f826ecc92425c8bac4d1ea69bcbbddaea4
            0x64, 0x72, 0xbf, 0x69, 0x2a, 0xaf, 0x27, 0x0d, 0x5f, 0x9d, 0xc4, 0x0c, 0x5e, 0xca,
            0xb8, 0xf8, 0x26, 0xec, 0xc9, 0x24, 0x25, 0xc8, 0xba, 0xc4, 0xd1, 0xea, 0x69, 0xbc,
            0xbb, 0xdd, 0xae, 0xa4,
        ];

        let input_2 = b"Hello, wgsl.\n";
        let expect_2: [u8; 32] = [
            193, 186, 14, 10, 195, 53, 238, 147, 57, 104, 6, 44, 255, 35, 108, 50, 166, 242, 19,
            147, 88, 218, 128, 198, 86, 91, 208, 2, 254, 200, 188, 56,
        ];

        let hashes = run_sha256_gpu(&[input_1, input_2]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expect_1, expect_2]);

        // Test again to ensure repeated calls work right.
        let hashes_again = run_sha256_gpu(&[input_1, input_2]).unwrap();
        assert_eq!(hashes_again.len(), 2);
        assert_eq!(hashes_again, vec![expect_1, expect_2]);
    }

    #[test]
    fn test_sha256_empty() {
        let input_1 = b"";
        let expect_1: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(sha256(input_1), expect_1);

        let hashes = run_sha256_gpu(&[input_1, input_1]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expect_1, expect_1]);
    }

    /// Same-length messages with `len(input) % 4 == 0` (aligned).
    #[test]
    fn test_sha256_is_valid_aligned_single_item() {
        let input_1 = b"Hello, wgsl\n";
        let expect_1: [u8; 32] = [
            254, 234, 146, 74, 232, 68, 234, 160, 191, 118, 232, 179, 211, 60, 233, 49, 144, 98,
            156, 231, 56, 159, 25, 217, 66, 189, 1, 131, 11, 167, 119, 180,
        ];

        let hashes = run_sha256_gpu(&[input_1]).unwrap();
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes, vec![expect_1]);
    }

    #[test]
    fn test_sha256_is_valid_69_byte_input() {
        let input_1 = b"Hello world. Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let expect_1: [u8; 32] = [
            177, 67, 166, 211, 213, 73, 234, 28, 0, 73, 118, 6, 242, 12, 47, 50, 118, 9, 161, 47,
            140, 0, 188, 98, 255, 175, 205, 243, 220, 243, 210, 168,
        ];

        let hashes = run_sha256_gpu(&[input_1]).unwrap();
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes, vec![expect_1]);
    }

    #[test]
    fn test_sha256d_in_steps() {
        let input_1 = b"Hello world.\n";
        let expect_1: [u8; 32] = [
            // 6472bf692aaf270d5f9dc40c5ecab8f826ecc92425c8bac4d1ea69bcbbddaea4
            0x64, 0x72, 0xbf, 0x69, 0x2a, 0xaf, 0x27, 0x0d, 0x5f, 0x9d, 0xc4, 0x0c, 0x5e, 0xca,
            0xb8, 0xf8, 0x26, 0xec, 0xc9, 0x24, 0x25, 0xc8, 0xba, 0xc4, 0xd1, 0xea, 0x69, 0xbc,
            0xbb, 0xdd, 0xae, 0xa4,
        ];

        let hashes_1 = run_sha256_gpu(&[input_1]).unwrap();
        assert_eq!(hashes_1.len(), 1);
        assert_eq!(hashes_1, vec![expect_1]);

        // Now, hash expect_1 and get expect_2.
        let expect_2: [u8; 32] = [
            184, 215, 246, 75, 181, 105, 139, 209, 5, 34, 213, 195, 67, 95, 62, 167, 203, 177, 223,
            133, 225, 14, 113, 253, 51, 66, 99, 113, 155, 48, 140, 144,
        ];

        let hashes_2 = run_sha256_gpu(&[&expect_1[..]]).unwrap();
        assert_eq!(hashes_2.len(), 1);
        assert_eq!(hashes_2, vec![expect_2]);
    }

    #[test]
    fn test_sha256d_is_valid() {
        let input = b"Hello world.\n";
        let expected: [u8; 32] = [
            184, 215, 246, 75, 181, 105, 139, 209, 5, 34, 213, 195, 67, 95, 62, 167, 203, 177, 223,
            133, 225, 14, 113, 253, 51, 66, 99, 113, 155, 48, 140, 144,
        ];

        let hashes = run_sha256d_gpu(&[&input[..], &input[..]]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expected, expected]);

        // Test again to ensure repeated calls work right.
        let hashes_again = run_sha256d_gpu(&[&input[..], &input[..]]).unwrap();
        assert_eq!(hashes_again.len(), 2);
        assert_eq!(hashes_again, vec![expected, expected]);
    }

    #[test]
    fn test_switching_pipeline_versions() {
        let input = b"Hello world.\n";
        let expected_sha256d: [u8; 32] = [
            184, 215, 246, 75, 181, 105, 139, 209, 5, 34, 213, 195, 67, 95, 62, 167, 203, 177, 223,
            133, 225, 14, 113, 253, 51, 66, 99, 113, 155, 48, 140, 144,
        ];
        let expected_sha256: [u8; 32] = [
            // 6472bf692aaf270d5f9dc40c5ecab8f826ecc92425c8bac4d1ea69bcbbddaea4
            0x64, 0x72, 0xbf, 0x69, 0x2a, 0xaf, 0x27, 0x0d, 0x5f, 0x9d, 0xc4, 0x0c, 0x5e, 0xca,
            0xb8, 0xf8, 0x26, 0xec, 0xc9, 0x24, 0x25, 0xc8, 0xba, 0xc4, 0xd1, 0xea, 0x69, 0xbc,
            0xbb, 0xdd, 0xae, 0xa4,
        ];

        let hashes_sha256d = run_sha256d_gpu(&[&input[..], &input[..]]).unwrap();
        assert_eq!(hashes_sha256d.len(), 2);
        assert_eq!(hashes_sha256d, vec![expected_sha256d, expected_sha256d]);

        let hashes_sha256 = run_sha256_gpu(&[&input[..], &input[..]]).unwrap();
        assert_eq!(hashes_sha256.len(), 2);
        assert_eq!(hashes_sha256, vec![expected_sha256, expected_sha256]);

        let hashes_sha256d = run_sha256d_gpu(&[&input[..], &input[..]]).unwrap();
        assert_eq!(hashes_sha256d.len(), 2);
        assert_eq!(hashes_sha256d, vec![expected_sha256d, expected_sha256d]);

        let hashes_sha256 = run_sha256_gpu(&[&input[..], &input[..]]).unwrap();
        assert_eq!(hashes_sha256.len(), 2);
        assert_eq!(hashes_sha256, vec![expected_sha256, expected_sha256]);
    }

    #[test]
    fn test_max_allowed_message_count_per_operation() {
        assert!(max_allowed_message_count_per_operation(64).unwrap() >= 256);
        assert!(max_allowed_message_count_per_operation(128).unwrap() >= 256);
        assert!(max_allowed_message_count_per_operation(4).unwrap() >= 256);
    }

    fn generate_random_array(size: usize, seed: u64) -> Vec<u8> {
        use rand::rngs::StdRng;
        use rand::{RngCore, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);

        let mut data = vec![0u8; size];
        rng.fill_bytes(&mut data);
        data
    }

    #[test]
    fn test_single_hash_increasing_lengths() {
        for input_len in 0..530 {
            let input_data = generate_random_array(input_len, input_len as u64);
            let expected = sha256(&input_data);
            if input_len == 0 {
                // Sanity check.
                assert_eq!(expected[0], 0xe3);
            }

            let hashes = run_sha256_gpu(&[&input_data[..input_len], &input_data[..]]).unwrap();
            assert_eq!(hashes.len(), 2);
            assert_eq!(
                hashes,
                vec![expected, expected],
                "Failure at input_len={}, input_data={:?}",
                input_len,
                input_data
            );
        }
    }

    #[test]
    fn test_sha256d_max_allowed_message_with_various_lengths() {
        if !get_test_config().enable_slow_tests {
            return;
        }

        // Increasing, then make sure to decrease size too to test under-utilized buffer case.
        let sizes = [0_usize, 1, 16, 55, 56, 65, 32];

        for input_len in sizes {
            let message_count: usize = max_allowed_message_count_per_operation(input_len).unwrap();
            // let message_count: usize = 10;
            println!(
                "Testing input_len={}, with message_count={}",
                input_len, message_count
            );

            let mut messages: Vec<Vec<u8>> = Vec::with_capacity(message_count);
            let mut expected: Vec<[u8; 32]> = Vec::with_capacity(message_count);
            for _message_num in 0..message_count {
                let input_data = generate_random_array(input_len, input_len as u64);

                expected.push(sha256d(&input_data)); // Note: This part takes longer than the GPU operation.
                messages.push(input_data);
            }

            let hashes: Vec<[u8; 32]> = run_sha256d_gpu(&messages).unwrap();
            assert_eq!(hashes.len(), message_count);
            std::hint::black_box(&hashes);

            assert!(
                hashes == expected, // Don't use assert_eq, otherwise the dump on fail is huge.
                "Failure: hashes != expected at input_len={}, message_count={}",
                input_len,
                message_count
            );
        }
    }

    #[test]
    fn test_single_hash_increasing_message_count() {
        let mut seed: u64 = 42;
        for (message_count_loop_num, message_count) in [1, 20, 50, 25, 1].iter().enumerate() {
            let mut input_data: Vec<Vec<u8>> = Vec::with_capacity(*message_count);

            for _message_num in 0..*message_count {
                input_data.push(generate_random_array(50, seed));
                seed += 1;
            }

            let expected: Vec<[u8; 32]> = input_data.iter().map(|x| sha256(x)).collect();

            let hashes: Vec<[u8; 32]> = run_sha256_gpu(&input_data).unwrap();

            assert_eq!(hashes.len(), *message_count);
            assert_eq!(
                hashes, expected,
                "Mismatch at message_count={}, message_count_loop_num={}",
                message_count, message_count_loop_num
            );
        }
    }

    #[test]
    fn test_calc_allowed_message_count_per_operation() {
        // Upper boundary case of single-block.
        assert_eq!(
            calc_allowed_message_count_per_operation(55, 256, 65_535, &[268435456, 134217728]),
            2_097_152
        );

        // Lower boundary case of two blocks. Caused issues.
        assert_eq!(
            calc_allowed_message_count_per_operation(56, 256, 65_535, &[268435456, 134217728]),
            1_048_576
        );
    }
}

// MARK: Test Pad

#[cfg(test)]
mod padding_tests {
    use super::*;

    #[test]
    fn test_calc_padded_message_length_bytes() {
        assert_eq!(calc_padded_message_length_bytes(0), 64);
        assert_eq!(calc_padded_message_length_bytes(20), 64);
        assert_eq!(calc_padded_message_length_bytes(31), 64);
        assert_eq!(calc_padded_message_length_bytes(32), 64);
        assert_eq!(calc_padded_message_length_bytes(35), 64);
        assert_eq!(calc_padded_message_length_bytes(50), 64);
        assert_eq!(calc_padded_message_length_bytes(55), 64);
        assert_eq!(calc_padded_message_length_bytes(56), 128);
        assert_eq!(calc_padded_message_length_bytes(57), 128);
        assert_eq!(calc_padded_message_length_bytes(64), 128);
        assert_eq!(calc_padded_message_length_bytes(65), 128);
    }

    #[test]
    fn pad_message_into_empty() {
        let msg: &[u8] = b"";

        const PADDED_LEN: u32 = calc_padded_message_length_bytes(0);
        let mut padded = [0xFFFFFFFF_u32; PADDED_LEN as usize / 4];
        pad_message_into(msg, &mut padded);

        // Inspect as raw bytes via u32 decomposition.
        let mut bytes = Vec::new();
        for w in &padded {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }

        // First byte is 0x80
        assert_eq!(bytes[0], 0x80);

        // Zero padding until length.
        for idx in 1..(bytes.len()) {
            assert_eq!(
                bytes[idx], 0,
                "Expected zero padding then zero value at index {}",
                idx
            );
        }

        // Length = 0 bits
        assert_eq!(&bytes[bytes.len() - 8..], &[0u8; 8]);
    }

    #[test]
    fn pad_message_into_unaligned() {
        let msg = b"abc";

        const PADDED_LEN: u32 = calc_padded_message_length_bytes(3);
        let mut padded = [0xFFFFFFFF_u32; PADDED_LEN as usize / 4];
        pad_message_into(msg, &mut padded[..]);

        let mut bytes = Vec::new();
        for w in &padded {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }

        // Message copied
        assert_eq!(&bytes[..3], msg);

        // Padding bit
        assert_eq!(bytes[3], 0x80);

        // Zero padding until length.
        for idx in 4..(bytes.len() - 8) {
            assert_eq!(bytes[idx], 0, "Expected zero padding at index {}", idx);
        }

        // Length = 24 bits (big endian)
        let bit_len = (msg.len() as u64) * 8;
        assert_eq!(&bytes[bytes.len() - 8..], &bit_len.to_be_bytes());
    }

    /// Test 3: Boundary case - 55 bytes (fits in one block)
    #[test]
    fn pad_message_into_55_bytes() {
        let msg = vec![0xAA; 55];
        let padded_len = calc_padded_message_length_bytes(msg.len() as u32);
        assert_eq!(padded_len, 64);

        const PADDED_LEN: u32 = calc_padded_message_length_bytes(55);
        let mut padded = [0xFFFFFFFF_u32; PADDED_LEN as usize / 4];
        pad_message_into(&msg, &mut padded[..]);

        let mut bytes = Vec::new();
        for w in &padded {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }

        // Message copied
        assert_eq!(&bytes[..55], &msg[..]);

        // Padding byte
        assert_eq!(bytes[55], 0x80);

        // Length immediately follows
        let bit_len = (msg.len() as u64) * 8;
        assert_eq!(&bytes[56..64], &bit_len.to_be_bytes());
    }

    /// Boundary case (56 bytes forces a second block).
    #[test]
    fn pad_message_into_56_bytes_two_blocks() {
        let msg = vec![0xBB; 56];

        const PADDED_LEN: u32 = calc_padded_message_length_bytes(56);
        let mut padded = [0xFFFFFFFF_u32; PADDED_LEN as usize / 4];
        pad_message_into(&msg, &mut padded[..]);
        assert_eq!(PADDED_LEN, 128);

        let mut bytes = Vec::new();
        for w in &padded {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }

        // Message copied
        assert_eq!(&bytes[..56], &msg[..]);

        // Padding byte
        assert_eq!(bytes[56], 0x80);

        // Zero padding until length.
        for idx in 57..(bytes.len() - 8) {
            assert_eq!(bytes[idx], 0, "Expected zero padding at index {}", idx);
        }

        // Length field
        let bit_len = (msg.len() as u64) * 8;
        assert_eq!(&bytes[bytes.len() - 8..], &bit_len.to_be_bytes());
    }

    // Test 5: Word packing sanity (no casts)
    #[test]
    fn pad_message_into_word_byte_order() {
        let msg = b"abcd";

        const PADDED_LEN: u32 = calc_padded_message_length_bytes(4);

        let mut padded = [0xFFFFFFFF_u32; PADDED_LEN as usize / 4];
        pad_message_into(msg, &mut padded[..]);

        let w = padded[0];
        let b = w.to_ne_bytes();

        assert_eq!(&b[..4], b"abcd");
    }
}
