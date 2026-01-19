// https://sotrh.github.io/learn-wgpu/compute/introduction/

use once_cell::sync::OnceCell;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt as _;

/// Calculate the length of the padded message.
///
/// Args:
/// - `len_bytes`: Length of the input message (before any padding).
///
/// Returns the padded length of the message.
fn calc_padded_message_length_bytes(len_bytes: u32) -> u32 {
    let len_bits = len_bytes * 8;

    // Number of bits to add so that:
    // len + 1 + k + 64 ≡ 0 (mod 512)
    let k = (512 - ((len_bits + 1 + 64) % 512)) % 512;
    let total_bits = len_bits + 1 + k + 64;

    debug_assert_eq!(total_bits % 512, 0);

    let total_bytes_with_padding = total_bits / 8;

    debug_assert_eq!(total_bytes_with_padding % 4, 0);
    debug_assert_eq!(total_bytes_with_padding % 64, 0); // SHA-256 invariant

    total_bytes_with_padding
}

/// Pad a message for SHA256 specifically (including appending its length).
///
/// Returns a Vec of length `padded_len_bytes`.
fn pad_message(bytes: &[u8], padded_len_bytes: u32) -> Vec<u32> {
    debug_assert!(
        padded_len_bytes % 4 == 0,
        "padded_len_bytes must be a multiple of 4"
    );
    debug_assert_eq!(padded_len_bytes % 64, 0);
    debug_assert_eq!(
        padded_len_bytes,
        calc_padded_message_length_bytes(bytes.len() as u32)
    );

    let padded_len_words = (padded_len_bytes / 4) as usize;
    let mut output_words = vec![0u32; padded_len_words];

    let buf = bytemuck::cast_slice_mut::<u32, u8>(&mut output_words);

    // Copy message.
    buf[..bytes.len()].copy_from_slice(bytes);

    // Append the 0x80 bit.
    buf[bytes.len()] = 0x80;

    // Append message length in bits (big endian).
    let bit_len = (bytes.len() as u64) * 8;
    let len_pos = buf.len() - 8;
    buf[len_pos..].copy_from_slice(&bit_len.to_be_bytes());

    // Convert to u32 words.
    debug_assert!(output_words.len() as u32 == (padded_len_bytes / 4));

    output_words
}

fn calc_num_workgroups(device: &wgpu::Device, num_messages: usize) -> u32 {
    let max_wg = device.limits().max_compute_workgroup_size_x;
    let max_groups = device.limits().max_compute_workgroups_per_dimension;

    let num = ((num_messages as u32) + max_wg - 1) / max_wg;
    if num > max_groups {
        panic!("Input array too large. Max size is {}", max_groups * max_wg);
    }
    num
}

fn validate_messages(messages: &[&[u8]]) {
    let len = messages[0].len();
    for m in messages {
        if m.len() != len {
            panic!("Messages must have the same size");
        }
    }
}

struct GPU {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
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

        // Inject configuration values into sha256-gpu.
        let pipeline_constants: &[(&str, f64)] = &[(
            &"CONFIG_WORKGROUP_SIZE",
            device.limits().max_compute_workgroup_size_x as f64,
        )];

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SHA256 Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("sha256"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: pipeline_constants,
                ..Default::default()
            },
            cache: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            compute_pipeline,
        })
    }
}

static GPU_INSTANCE: OnceCell<GPU> = OnceCell::new();
fn get_gpu() -> anyhow::Result<&'static GPU> {
    GPU_INSTANCE.get_or_try_init(|| pollster::block_on(GPU::init()))
}

pub async fn sha256_gpu(messages: &[&[u8]]) -> anyhow::Result<Vec<[u8; 32]>> {
    validate_messages(&messages);

    let gpu = get_gpu()?;

    let num_messages = messages.len();
    let num_workgroups = calc_num_workgroups(&gpu.device, num_messages);

    let message_length = messages[0].len();
    let padded_message_length_bytes = calc_padded_message_length_bytes(message_length as u32);

    // ---- pack messages ----
    let mut message_array = Vec::<u32>::new();
    for msg in messages {
        let padded = pad_message(msg, padded_message_length_bytes);
        message_array.extend_from_slice(&padded);
    }

    let message_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("messages"),
            contents: bytemuck::cast_slice(&message_array),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let num_messages_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("num_messages"),
            contents: bytemuck::cast_slice(&[num_messages as u32]),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // TODO: Maybe pass these in as two separate inputs.
    // `message_sizes[0]` is the original message length in bytes.
    // `message_sizes[1]` is the padded message length in bytes.
    let message_sizes: [u32; 2] = [message_length as u32, padded_message_length_bytes];
    let message_sizes_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("message_sizes"),
            contents: bytemuck::cast_slice(&message_sizes),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // ---- result buffer (still contiguous on GPU) ----
    let result_buffer_size = (32 * num_messages) as u64;

    let result_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hashes"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &gpu.compute_pipeline.get_bind_group_layout(0),
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
        label: None,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&gpu.compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback, 0, result_buffer_size);
    gpu.queue.submit(Some(encoder.finish()));

    // ---- readback ----
    // readback.map_async(wgpu::MapMode::Read).await?;
    // let mapped = readback.slice(..).get_mapped_range();

    let buffer_slice = readback.slice(..);

    let map_done = Arc::new(Mutex::new(None));
    let map_done_clone = map_done.clone();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        *map_done_clone.lock().unwrap() = Some(result);
    });

    // VERY IMPORTANT: drive the mapping to completion
    // gpu.device.poll(wgpu::Maintain::Wait);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

    // Check mapping result
    match map_done.lock().unwrap().take().unwrap() {
        Ok(()) => {}
        Err(e) => return Err(e.into()),
    }

    let mapped = buffer_slice.get_mapped_range();

    // Reshape from long output array into separate hash values for each message.
    let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(num_messages);
    for i in 0..num_messages {
        let start = i * 32;
        let end = start + 32;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&mapped[start..end]);
        hashes.push(hash);
    }

    drop(mapped);
    readback.unmap();

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
    use std::cmp::{max, min};

    // Based on logic/docs in `calc_num_workgroups()`.
    let max_from_workgroups = (device_max_wg_per_dimension * device_max_wg_size_x) as usize;

    // Based on errors about max buffer size.
    let each_message_length_64_bits = max(each_message_length, 64); // Maybe need padding logic.
    let device_max_buffer_size = *device_max_buffer_sizes
        .iter()
        .min() // Pick the tighter option.
        .unwrap(); // Unwrap is safe because the array should never be empty.
    let max_from_buffer = (device_max_buffer_size / each_message_length_64_bits as u64) as usize;

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

pub fn run_sha256_gpu(messages: &[&[u8]]) -> anyhow::Result<Vec<[u8; 32]>> {
    pollster::block_on(sha256_gpu(messages))
}

// MARK: Tests

#[cfg(test)]
mod tests {
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

        let hashes = run_sha256_gpu(&[&input_1[..], &input_2[..]]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expect_1, expect_2]);

        // Test again to ensure repeated calls work right.
        let hashes_again = run_sha256_gpu(&[&input_1[..], &input_2[..]]).unwrap();
        assert_eq!(hashes_again.len(), 2);
        assert_eq!(hashes_again, vec![expect_1, expect_2]);
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

        let hashes = run_sha256_gpu(&[&input_1[..], &input_2[..]]).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes, vec![expect_1, expect_2]);

        // Test again to ensure repeated calls work right.
        let hashes_again = run_sha256_gpu(&[&input_1[..], &input_2[..]]).unwrap();
        assert_eq!(hashes_again.len(), 2);
        assert_eq!(hashes_again, vec![expect_1, expect_2]);
    }

    #[test]
    fn test_max_allowed_message_count_per_operation() {
        assert!(max_allowed_message_count_per_operation(64).unwrap() >= 256);
        assert!(max_allowed_message_count_per_operation(128).unwrap() >= 256);
        assert!(max_allowed_message_count_per_operation(4).unwrap() >= 256);
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
    fn pad_message_empty() {
        let msg: &[u8] = b"";
        let padded_len = calc_padded_message_length_bytes(0);
        let padded = pad_message(msg, padded_len);

        assert_eq!(padded.len() * 4, padded_len as usize);

        // Inspect as raw bytes via u32 decomposition
        let mut bytes = Vec::new();
        for w in &padded {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }

        // First byte is 0x80
        assert_eq!(bytes[0], 0x80);

        // All bytes except last 8 are zero
        for &b in &bytes[1..bytes.len() - 8] {
            assert_eq!(b, 0);
        }

        // Length = 0 bits
        assert_eq!(&bytes[bytes.len() - 8..], &[0u8; 8]);
    }

    #[test]
    fn pad_message_unaligned() {
        let msg = b"abc";
        let padded_len = calc_padded_message_length_bytes(msg.len() as u32);
        let padded = pad_message(msg, padded_len);

        let mut bytes = Vec::new();
        for w in &padded {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }

        // Message copied
        assert_eq!(&bytes[..3], msg);

        // Padding bit
        assert_eq!(bytes[3], 0x80);

        // Zero padding until length
        for &b in &bytes[4..bytes.len() - 8] {
            assert_eq!(b, 0);
        }

        // Length = 24 bits (big endian)
        let bit_len = (msg.len() as u64) * 8;
        assert_eq!(&bytes[bytes.len() - 8..], &bit_len.to_be_bytes());
    }

    /// Test 3: Boundary case - 55 bytes (fits in one block)
    #[test]
    fn pad_message_55_bytes() {
        let msg = vec![0xAA; 55];
        let padded_len = calc_padded_message_length_bytes(msg.len() as u32);
        assert_eq!(padded_len, 64);

        let padded = pad_message(&msg, padded_len);

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

    /// Boundary case (forces a second block).
    #[test]
    fn pad_message_56_bytes_two_blocks() {
        let msg = vec![0xBB; 56];
        let padded_len = calc_padded_message_length_bytes(msg.len() as u32);
        assert_eq!(padded_len, 128);

        let padded = pad_message(&msg, padded_len);

        let mut bytes = Vec::new();
        for w in &padded {
            bytes.extend_from_slice(&w.to_ne_bytes());
        }

        // Message copied
        assert_eq!(&bytes[..56], &msg[..]);

        // Padding byte
        assert_eq!(bytes[56], 0x80);

        // Zero padding until length
        for &b in &bytes[57..bytes.len() - 8] {
            assert_eq!(b, 0);
        }

        // Length field
        let bit_len = (msg.len() as u64) * 8;
        assert_eq!(&bytes[bytes.len() - 8..], &bit_len.to_be_bytes());
    }

    // Test 5: Word packing sanity (no casts)
    #[test]
    fn pad_message_word_byte_order() {
        let msg = b"abcd";
        let padded_len = calc_padded_message_length_bytes(4);
        let padded = pad_message(msg, padded_len);

        let w = padded[0];
        let b = w.to_ne_bytes();

        assert_eq!(&b[..4], b"abcd");
    }
}
