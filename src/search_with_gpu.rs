use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use log::info;

use crate::sha256_gpu::{max_allowed_message_count_per_operation, run_sha256_gpu};
use crate::types::{ChecksumPatternMatch, ChecksumPatternSpec};

pub fn search_for_checksums_gpu(
    data: &[u8],
    checksum_patterns: &[ChecksumPatternSpec],
) -> anyhow::Result<Vec<ChecksumPatternMatch>> {
    // Create progress bar.
    let progress_bar = ProgressBar::new(data.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({binary_bytes_per_sec}) - ETA {eta}")
            .unwrap()
            .progress_chars("#>-"),
    );
    if data.len() < 5_000_000 {
        // Hide the progress bar on small datasets. Important for keeping test output clean.
        progress_bar.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut matches: Vec<ChecksumPatternMatch> = Vec::new();

    let max_messages_per_operation = max_allowed_message_count_per_operation()?;
    info!(
        "GPU max messages per operation: {}",
        max_messages_per_operation
    );

    // First, we loop over patterns, because all messages in a GPU request must have the same length.
    for pattern in checksum_patterns {
        // Read in every pattern-chosen chunk to hash.
        let mut message_start_idx: usize = 0;
        let mut message_selection: Vec<&[u8]> = Vec::with_capacity(max_messages_per_operation);
        while message_start_idx < data.len() - pattern.total_length() {
            // Store the start index if the first element in this batch.
            let these_messages_start_idx = message_start_idx;

            // Fill the message buffer to send in bulk.
            while (message_start_idx < data.len() - pattern.total_length())
                && (message_selection.len() < max_messages_per_operation)
            {
                message_selection
                    .push(&data[message_start_idx..message_start_idx + pattern.chunk_len]);

                message_start_idx += 1;
            }

            // Send the buffer for hashing.
            let hash_results: Vec<[u8; 32]> = run_sha256_gpu(&message_selection)?;

            assert_eq!(hash_results.len(), message_selection.len());

            // FIXME: Need to double-hash here.

            // Check for any matches. Log and push.
            for message_num in 0..hash_results.len() {
                let chunk_start_idx = these_messages_start_idx + message_num;

                let checksum_in_data: &[u8] = &data
                    [chunk_start_idx + pattern.chunk_len..chunk_start_idx + pattern.total_length()];

                if checksum_in_data == hash_results[message_num] {
                    // This condition is very rare, and is the success case!
                    let chunk_and_checksum =
                        &data[chunk_start_idx..chunk_start_idx + pattern.total_length()];

                    info!(
                        "✅ Match! Offset: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}, Hash: {:x?}",
                        chunk_start_idx,
                        chunk_start_idx,
                        pattern.chunk_len,
                        chunk_and_checksum[..pattern.chunk_len].to_vec(),
                        hash_results[message_num]
                    );

                    matches.push(ChecksumPatternMatch {
                        chunk_len: pattern.chunk_len,
                        checksum_len: pattern.checksum_len,
                        chunk_start_offset: chunk_start_idx as u64,
                        chunk_data: chunk_and_checksum[..pattern.chunk_len].to_vec(),
                        checksum_data: chunk_and_checksum
                            [pattern.chunk_len..pattern.total_length()]
                            .to_vec(),
                    });
                }
            }

            // Clear the message selection for the next batch.
            message_selection.clear();
        }
    }

    progress_bar.finish_with_message("Search complete");

    info!("Search complete. Found {} matches.", matches.len());

    // Ensure they're in sorted order after Rayon potentially re-ordered them.
    matches.sort_by_key(|m| m.chunk_start_offset);

    info!("Matches sorted by chunk_start_offset.");

    Ok(matches)
}
