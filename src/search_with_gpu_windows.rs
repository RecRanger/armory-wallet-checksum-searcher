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
    info!(
        "Sending up to {} bytes ({:.3} MiB) to the GPU at once.",
        max_input_len_bytes,
        max_input_len_bytes as f64 / (1024.0 * 1024.0)
    );

    let mut matches: Vec<ChecksumPatternMatch> = Vec::new();

    // Loop through chunks, making sure to include overlap so boundary-spanning
    // matches are not missed.
    let longest_pattern_length_bytes = checksum_patterns
        .iter()
        .map(|p| p.total_length())
        .max()
        .unwrap_or(0);
    let overlap_len_bytes = longest_pattern_length_bytes.saturating_sub(1);
    let mut chunk_base_offset: usize = 0;

    while chunk_base_offset < data.len() {
        let chunk_end_exclusive =
            usize::min(chunk_base_offset + max_input_len_bytes as usize, data.len());

        let chunk_slice = &data[chunk_base_offset..chunk_end_exclusive];

        // One GPU invocation scans *this chunk* for this pattern.
        let matching_offsets_in_chunk = search_sha256_gpu_windows(
            chunk_slice,
            checksum_patterns,
            chunk_base_offset,
            ComputePipelineVersionWindows::Sha256DoubleWindows,
        )?;
        for checksum_pattern_match in matching_offsets_in_chunk {
            matches.push(checksum_pattern_match);
        }

        // Advance chunk base, keeping overlap.
        if chunk_end_exclusive == data.len() {
            break;
        }

        chunk_base_offset = chunk_end_exclusive - overlap_len_bytes;

        // Progress update:
        // Progress is measured over the *entire data buffer*, divided over patterns.
        progress_bar.set_position(chunk_base_offset as u64);
    }

    progress_bar.finish_with_message("Search complete");

    info!("Search complete. Found {} matches.", matches.len());

    // Ensure deterministic order.
    matches.sort_by_key(|m| m.chunk_start_offset);

    // Remove perfect duplicate elements in matches. Must be sorted first.
    // Important because shorter patterns near boundaries may sneak into two searches.
    matches.dedup();

    info!("Matches sorted by chunk_start_offset.");

    Ok(matches)
}
