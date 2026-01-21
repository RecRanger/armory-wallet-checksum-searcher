use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::info;

use crate::sha256_gpu_windows::{
    ComputePipelineVersionWindows, get_max_input_len_bytes, search_sha256_gpu_windows,
};
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

    // Look up the GPU's max buffer size.
    let max_input_len_bytes = get_max_input_len_bytes()?;

    let mut matches: Vec<ChecksumPatternMatch> = Vec::new();

    // We iterate patterns outermost to keep GPU config uniform per dispatch.
    for (pattern_idx, pattern) in checksum_patterns.iter().enumerate() {
        // Edge case: data too small for this pattern
        if data.len() < pattern.total_length() {
            continue;
        }

        info!(
            "GPU search: pattern_idx={}, chunk_len={}, checksum_len={}, total_len={}",
            pattern_idx,
            pattern.chunk_len,
            pattern.checksum_len,
            pattern.total_length()
        );

        // Loop through chunks, making sure to include overlap so boundary-spanning
        // matches are not missed.
        let overlap_len_bytes = pattern.total_length().saturating_sub(1);
        let mut chunk_base_offset: usize = 0;

        while chunk_base_offset < data.len() {
            let chunk_end_exclusive =
                usize::min(chunk_base_offset + max_input_len_bytes as usize, data.len());

            let chunk_slice = &data[chunk_base_offset..chunk_end_exclusive];

            // Skip chunks that are too small to possibly contain a full pattern.
            // Rare edge-case in the very last chunk.
            if chunk_slice.len() < pattern.total_length() {
                break;
            }

            // One GPU invocation scans *this chunk* for this pattern.
            let matching_offsets_in_chunk = search_sha256_gpu_windows(
                chunk_slice,
                pattern.chunk_len as u32,
                pattern.checksum_len as u32,
                ComputePipelineVersionWindows::Sha256DoubleWindows,
            )?;

            // Materialize matches (rare path).
            for &relative_offset in &matching_offsets_in_chunk {
                let absolute_offset = chunk_base_offset + relative_offset as usize;

                let chunk_and_checksum =
                    &data[absolute_offset..absolute_offset + pattern.total_length()];

                let chunk_data = &chunk_and_checksum[..pattern.chunk_len];
                let checksum_data = &chunk_and_checksum[pattern.chunk_len..pattern.total_length()];

                debug_assert_eq!(chunk_data.len(), pattern.chunk_len);
                debug_assert_eq!(checksum_data.len(), pattern.checksum_len);

                info!(
                    "✅ Match! Offset: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}",
                    absolute_offset, absolute_offset, pattern.chunk_len, chunk_data
                );

                matches.push(ChecksumPatternMatch {
                    chunk_len: pattern.chunk_len,
                    checksum_len: pattern.checksum_len,
                    chunk_start_offset: absolute_offset as u64,
                    chunk_data: chunk_data.to_vec(),
                    checksum_data: checksum_data.to_vec(),
                });
            }

            // Advance chunk base, keeping overlap.
            if chunk_end_exclusive == data.len() {
                break;
            }

            chunk_base_offset = chunk_end_exclusive - overlap_len_bytes;

            // Progress update:
            // Progress is measured over the *entire data buffer*, divided over patterns.
            progress_bar.set_position(
                ((
                    // Fully done patterns.
                    (data.len() * pattern_idx)
                    // Plus the fraction of this pattern.
                    + chunk_base_offset
                ) / checksum_patterns.len()) as u64,
            );
        }
    }

    progress_bar.finish_with_message("Search complete");

    info!("Search complete. Found {} matches.", matches.len());

    // Ensure deterministic order.
    matches.sort_by_key(|m| m.chunk_start_offset);

    info!("Matches sorted by chunk_start_offset.");

    Ok(matches)
}
