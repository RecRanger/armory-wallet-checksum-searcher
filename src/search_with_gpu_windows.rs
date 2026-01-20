use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::info;

use crate::sha256_gpu_windows::{ComputePipelineVersionWindows, search_sha256_gpu_windows};
use crate::types::{ChecksumPatternMatch, ChecksumPatternSpec};

pub fn search_for_checksums_gpu_windows(
    data: &[u8],
    checksum_patterns: &[ChecksumPatternSpec],
) -> anyhow::Result<Vec<ChecksumPatternMatch>> {
    // Create progress bar.
    let progress_bar = ProgressBar::new(data.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(
                "GPU: {spinner:.green} [{elapsed_precise}] \
                 [{bar:40.cyan/blue}] {bytes}/{total_bytes} \
                 ({binary_bytes_per_sec}) - ETA {eta}",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    if data.len() < 5_000_000 {
        // Hide the progress bar on small datasets. Important for keeping test output clean.
        progress_bar.set_draw_target(ProgressDrawTarget::hidden());
    }

    progress_bar.set_position(0);

    let mut matches: Vec<ChecksumPatternMatch> = Vec::new();

    // We iterate patterns outermost to keep GPU config uniform per dispatch.
    for (pattern_idx, pattern) in checksum_patterns.iter().enumerate() {
        let message_len = pattern.chunk_len as u32;
        let compare_len = pattern.checksum_len as u32;

        // Edge case: data too small for this pattern
        if data.len() < pattern.total_length() {
            continue;
        }

        info!(
            "GPU search: chunk_len={}, checksum_len={}, total_len={}",
            pattern.chunk_len,
            pattern.checksum_len,
            pattern.total_length()
        );

        // One GPU invocation scans the entire buffer for this pattern.
        let matching_offsets = search_sha256_gpu_windows(
            data,
            message_len,
            compare_len,
            ComputePipelineVersionWindows::Sha256DoubleWindows,
        )?;

        // Materialize matches (rare path).
        for &chunk_start_idx in &matching_offsets {
            let chunk_start_idx = chunk_start_idx as usize;

            let chunk_and_checksum =
                &data[chunk_start_idx..chunk_start_idx + pattern.total_length()];

            let chunk_data = &chunk_and_checksum[..pattern.chunk_len];
            let checksum_data = &chunk_and_checksum[pattern.chunk_len..pattern.total_length()];

            debug_assert_eq!(chunk_data.len(), pattern.chunk_len);
            debug_assert_eq!(checksum_data.len(), pattern.checksum_len);

            info!(
                "✅ Match! Offset: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}",
                chunk_start_idx, chunk_start_idx, pattern.chunk_len, chunk_data
            );

            matches.push(ChecksumPatternMatch {
                chunk_len: pattern.chunk_len,
                checksum_len: pattern.checksum_len,
                chunk_start_offset: chunk_start_idx as u64,
                chunk_data: chunk_data.to_vec(),
                checksum_data: checksum_data.to_vec(),
            });
        }

        // Progress update:
        // Progress is measured over the *entire data buffer*, amortized over patterns.
        progress_bar
            .set_position(((data.len() * (pattern_idx + 1)) / checksum_patterns.len()) as u64);
    }

    progress_bar.finish_with_message("Search complete");

    info!("Search complete. Found {} matches.", matches.len());

    // Ensure deterministic order.
    matches.sort_by_key(|m| m.chunk_start_offset);

    info!("Matches sorted by chunk_start_offset.");

    Ok(matches)
}
