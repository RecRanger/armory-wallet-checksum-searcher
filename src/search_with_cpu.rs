use indicatif::{ParallelProgressIterator as _, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use log::info;

use crate::types::{ChecksumPatternMatch, ChecksumPatternSpec};

#[allow(unused)] // Helpful in tests.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

pub fn sha256d(data: &[u8]) -> [u8; 32] {
    // Basic: sha256(&sha256(data))
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result_1 = hasher.finalize();

    // Second round.
    let mut hasher = Sha256::new();
    hasher.update(&result_1);
    hasher.finalize().into()
}

#[inline]
pub fn compute_checksum(data: &[u8]) -> [u8; 32] {
    sha256d(data)
}

/// Find all checksum pattern matches at a given offset in the data.
///
/// Searches all pattern specs, starting from the chunk_start_idx.
fn find_matches_at_offset(
    data: &[u8],
    checksum_patterns: &[ChecksumPatternSpec],
    chunk_start_idx: usize,
    max_checksum_pattern_total_length: usize,
) -> Vec<ChecksumPatternMatch> {
    // This function _could_ return multiple matches (as many tests assert) with subset patterns.
    // Create a local Vec to make it possible to return many matches.
    // Performance: Constructing this empty Vec does not result in an allocation performance
    // penalty. The allocation cost is only paid on insertion (rare).
    let mut local_matches = Vec::new();

    // Skip if chunk and checksum are all zeros (check once for longest pattern only).
    // First, safety check for right at the end of the dataset, avoid array overruns.
    if chunk_start_idx + max_checksum_pattern_total_length < data.len()
        && data[chunk_start_idx..(chunk_start_idx + max_checksum_pattern_total_length)]
            .iter()
            .all(|&x| x == 0)
    {
        return local_matches;
    }

    for pattern in checksum_patterns {
        if chunk_start_idx + pattern.total_length() > data.len() {
            continue;
        }

        let chunk_and_checksum = &data[chunk_start_idx..chunk_start_idx + pattern.total_length()];

        let hash_result = compute_checksum(&chunk_and_checksum[..pattern.chunk_len]);

        // Success criteria: Hash output matches.
        if hash_result[..pattern.checksum_len] == chunk_and_checksum[pattern.chunk_len..] {
            info!(
                "✅ Match! Offset: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}, Hash: {:x?}",
                chunk_start_idx,
                chunk_start_idx,
                pattern.chunk_len,
                chunk_and_checksum[..pattern.chunk_len].to_vec(),
                hash_result
            );

            local_matches.push(ChecksumPatternMatch {
                chunk_len: pattern.chunk_len,
                checksum_len: pattern.checksum_len,
                chunk_start_offset: chunk_start_idx as u64,
                chunk_data: chunk_and_checksum[..pattern.chunk_len].to_vec(),
                checksum_data: chunk_and_checksum[pattern.chunk_len..pattern.total_length()]
                    .to_vec(),
            });
        }
    }

    local_matches
}

pub fn search_for_checksums_cpu(
    data: &[u8],
    checksum_patterns: &[ChecksumPatternSpec],
) -> Vec<ChecksumPatternMatch> {
    // Create progress bar.
    let progress_bar = ProgressBar::new(data.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("CPU: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({binary_bytes_per_sec}) - ETA {eta}")
            .unwrap()
            .progress_chars("#>-"),
    );
    if data.len() < 5_000_000 {
        // Hide the progress bar on small datasets. Important for keeping test output clean.
        progress_bar.set_draw_target(ProgressDrawTarget::hidden());
    }

    let max_checksum_pattern_total_length: usize = checksum_patterns
        .iter()
        .map(|pattern| pattern.total_length())
        .max()
        .unwrap_or(0);

    // Iterate over every chunk starting point, with parallel workers for each starting point.
    let mut matches: Vec<ChecksumPatternMatch> = (0..data.len())
        .into_par_iter()
        .progress_with(progress_bar.clone())
        .flat_map(|chunk_start_idx| {
            find_matches_at_offset(
                data,
                checksum_patterns,
                chunk_start_idx,
                max_checksum_pattern_total_length,
            )
        })
        .collect();

    progress_bar.finish_with_message("Search complete");

    info!("Search complete. Found {} matches.", matches.len());

    // Ensure they're in sorted order after Rayon potentially re-ordered them.
    matches.sort_by_key(|m| m.chunk_start_offset);

    info!("Matches sorted by chunk_start_offset.");

    matches
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_is_valid() {
        let input_1 = b"";
        let expect_1: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(sha256(input_1), expect_1);

        let input_2 = b"Hello world.\n";
        let expect_2: [u8; 32] = [
            // 6472bf692aaf270d5f9dc40c5ecab8f826ecc92425c8bac4d1ea69bcbbddaea4
            0x64, 0x72, 0xbf, 0x69, 0x2a, 0xaf, 0x27, 0x0d, 0x5f, 0x9d, 0xc4, 0x0c, 0x5e, 0xca,
            0xb8, 0xf8, 0x26, 0xec, 0xc9, 0x24, 0x25, 0xc8, 0xba, 0xc4, 0xd1, 0xea, 0x69, 0xbc,
            0xbb, 0xdd, 0xae, 0xa4,
        ];
        assert_eq!(sha256(input_2), expect_2);

        let input_3 = b"Hello, wgsl.\n";
        let expect_3: [u8; 32] = [
            193, 186, 14, 10, 195, 53, 238, 147, 57, 104, 6, 44, 255, 35, 108, 50, 166, 242, 19,
            147, 88, 218, 128, 198, 86, 91, 208, 2, 254, 200, 188, 56,
        ];
        assert_eq!(sha256(input_3), expect_3);

        let input = b"Hello, wgsl\n";
        let expect: [u8; 32] = [
            254, 234, 146, 74, 232, 68, 234, 160, 191, 118, 232, 179, 211, 60, 233, 49, 144, 98,
            156, 231, 56, 159, 25, 217, 66, 189, 1, 131, 11, 167, 119, 180,
        ];
        assert_eq!(sha256(input), expect);

        let input = b"Hello world\n";
        let expect: [u8; 32] = [
            24, 148, 161, 156, 133, 186, 21, 58, 203, 247, 67, 172, 78, 67, 252, 0, 76, 137, 22, 4,
            178, 111, 140, 105, 225, 232, 62, 162, 175, 199, 196, 143,
        ];
        assert_eq!(sha256(input), expect);
    }

    #[test]
    fn test_sha256d_is_valid() {
        let input_1 = b"";
        let expect_1: [u8; 32] = [
            93, 246, 224, 226, 118, 19, 89, 211, 10, 130, 117, 5, 142, 41, 159, 204, 3, 129, 83,
            69, 69, 245, 92, 244, 62, 65, 152, 63, 93, 76, 148, 86,
        ];
        assert_eq!(sha256d(input_1), expect_1);

        let input_2 = b"Hello world.\n";
        let expect_2: [u8; 32] = [
            184, 215, 246, 75, 181, 105, 139, 209, 5, 34, 213, 195, 67, 95, 62, 167, 203, 177, 223,
            133, 225, 14, 113, 253, 51, 66, 99, 113, 155, 48, 140, 144,
        ];
        assert_eq!(sha256d(input_2), expect_2);
    }
}
