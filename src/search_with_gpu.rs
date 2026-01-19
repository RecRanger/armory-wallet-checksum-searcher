use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use log::info;

use crate::search_with_cpu::sha256d;
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
            .template("GPU: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({binary_bytes_per_sec}) - ETA {eta}")
            .unwrap()
            .progress_chars("#>-"),
    );
    if data.len() < 5_000_000 {
        // Hide the progress bar on small datasets. Important for keeping test output clean.
        progress_bar.set_draw_target(ProgressDrawTarget::hidden());
    }
    progress_bar.set_position(0); // Set to 0 at the start to show it.

    let mut matches: Vec<ChecksumPatternMatch> = Vec::new();

    // First, we loop over patterns, because all messages in a GPU request must have the same length.
    for (pattern_idx, pattern) in checksum_patterns.iter().enumerate() {
        // Determine max number of messages in each group.
        let max_messages_per_operation =
            max_allowed_message_count_per_operation(pattern.chunk_len)?;
        info!(
            "GPU max messages per operation (for chunk_len={}): {}",
            pattern.chunk_len, max_messages_per_operation
        );

        // Silly edge case (useful for tests).
        if data.len() < pattern.total_length() {
            continue;
        }

        // Read in every pattern-chosen chunk to hash.
        let mut message_start_idx: usize = 0;
        let mut message_selection: Vec<&[u8]> = Vec::with_capacity(max_messages_per_operation);
        while message_start_idx <= data.len() - pattern.total_length() {
            // Store the start index of the first element in this batch.
            let message_batch_start_offset = message_start_idx;

            // Fill the message buffer to send in bulk.
            while (message_start_idx <= data.len() - pattern.total_length())
                && (message_selection.len() < max_messages_per_operation)
            {
                let chunk_data = &data[message_start_idx..message_start_idx + pattern.chunk_len];
                debug_assert_eq!(chunk_data.len(), pattern.chunk_len);
                message_selection.push(chunk_data);

                message_start_idx += 1;
            }

            // Send the buffer for hashing.
            let hash_results: Vec<[u8; 32]> = run_sha256_gpu(&message_selection)?;

            debug_assert_eq!(hash_results.len(), message_selection.len());

            // TODO: Move the double-hash step into the shader to avoid martialling through RAM.

            let hash_results: Vec<[u8; 32]> = run_sha256_gpu(
                &hash_results
                    .iter()
                    .map(|h| h.as_ref())
                    .collect::<Vec<&[u8]>>(),
            )?;

            // CPU-based override for this logic style.
            // let hash_results: Vec<[u8; 32]> =
            //     message_selection.iter().map(|x| sha256d(x)).collect();

            // Critical premise.
            debug_assert_eq!(hash_results.len(), message_selection.len());

            // Check for any matches. Log and push.
            for message_num in 0..hash_results.len() {
                let hash_result = hash_results[message_num];

                let chunk_start_idx = message_batch_start_offset + message_num;

                debug_assert_eq!(
                    sha256d(message_selection[message_num]),
                    hash_results[message_num]
                );

                let chunk_and_checksum: &[u8] =
                    &data[chunk_start_idx..chunk_start_idx + pattern.total_length()];

                let checksum_in_data: &[u8] = &data
                    [chunk_start_idx + pattern.chunk_len..chunk_start_idx + pattern.total_length()];

                debug_assert_eq!(checksum_in_data.len(), pattern.checksum_len);
                debug_assert_eq!(chunk_and_checksum.len(), pattern.total_length());

                if hash_result[..pattern.checksum_len] == chunk_and_checksum[pattern.chunk_len..] {
                    // This condition is very rare, and is the success case!

                    info!(
                        "✅ Match! Offset: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}, Hash: {:x?}",
                        chunk_start_idx,
                        chunk_start_idx,
                        pattern.chunk_len,
                        chunk_and_checksum[..pattern.chunk_len].to_vec(),
                        hash_result
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

            // Update the progress bar. We want to total bytes size to show the total file size, so we
            // have to count the data from the completed patterns, and then divide by the number of patterns.
            progress_bar.set_position(
                (((data.len() * pattern_idx) + (message_start_idx)) / (checksum_patterns.len()))
                    as u64,
            );
        }
    }

    progress_bar.finish_with_message("Search complete");

    info!("Search complete. Found {} matches.", matches.len());

    // Ensure they're in sorted order after Rayon potentially re-ordered them.
    matches.sort_by_key(|m| m.chunk_start_offset);

    info!("Matches sorted by chunk_start_offset.");

    Ok(matches)
}
