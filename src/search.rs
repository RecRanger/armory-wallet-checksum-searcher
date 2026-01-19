use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use indexmap::IndexMap;
use memmap2::Mmap;

use log::info;

use crate::{
    search_with_cpu::search_for_checksums_cpu,
    search_with_gpu::search_for_checksums_gpu,
    types::{ChecksumPatternMatch, ChecksumPatternSpec, ProcessorChoice},
};

pub fn search_for_checksums(
    data: &[u8],
    checksum_patterns: &[ChecksumPatternSpec],
    processor_choice: ProcessorChoice,
) -> anyhow::Result<Vec<ChecksumPatternMatch>> {
    match processor_choice {
        ProcessorChoice::Cpu => Ok(search_for_checksums_cpu(data, checksum_patterns)),
        ProcessorChoice::Gpu => search_for_checksums_gpu(data, checksum_patterns),
    }
}

pub fn process_file(
    file_path: &PathBuf,
    checksum_patterns: &[ChecksumPatternSpec],
    processor_choice: ProcessorChoice,
) -> anyhow::Result<Vec<ChecksumPatternMatch>> {
    let file = File::open(file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Optimization: Inform the kernel that it's fine to dump old pages after we're past,
    // and that we'll be requesting forward-looking pages continuously.
    mmap.advise(memmap2::Advice::Sequential)?;

    info!(
        "Memory-mapped file: {} bytes ({:.3} GiB)",
        mmap.len(),
        (mmap.len() as f64 / 1024.0 / 1024.0 / 1024.0)
    );

    let start_time = Instant::now();

    // Search the entire file for every pattern. This is the main long operation.
    let pattern_matches = search_for_checksums(&mmap[..], checksum_patterns, processor_choice)?;

    // Log performance stats.
    let duration = start_time.elapsed();
    let data_rate_mebibytes_per_sec =
        (mmap.len() as f64) / duration.as_secs_f64() / (1024.0 * 1024.0);
    let hash_rate_hash_per_sec =
        (mmap.len() as f64) * (checksum_patterns.len() as f64) / duration.as_secs_f64();
    info!(
        "Completed in {:?} ({:.3} MiB/s, {:.3} MH/s)",
        duration,
        data_rate_mebibytes_per_sec,
        hash_rate_hash_per_sec / 1e6,
    );

    // Log results stats.
    let successes_per_pattern = count_successes_per_pattern(checksum_patterns, &pattern_matches);
    info!(
        "Total successes: {} = {:?}",
        pattern_matches.len(),
        successes_per_pattern
    );

    Ok(pattern_matches)
}

/// Count how many successes of each pattern type.
fn count_successes_per_pattern(
    checksum_patterns: &[ChecksumPatternSpec],
    pattern_matches: &Vec<ChecksumPatternMatch>,
) -> IndexMap<String, i32> {
    let mut checksum_pattern_success_count: IndexMap<String, i32> = IndexMap::new();
    for checksum_pattern in checksum_patterns {
        checksum_pattern_success_count.insert(checksum_pattern.to_string(), 0);
    }

    for pattern_match in pattern_matches {
        let pattern_str = pattern_match.to_checksum_pattern_string();
        let count = checksum_pattern_success_count
            .entry(pattern_str)
            .or_insert(0);
        *count += 1;
    }
    checksum_pattern_success_count
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use super::*;
    use rstest::rstest;

    use crate::search_with_cpu::compute_checksum;
    use crate::test_general::get_test_config;
    use crate::types::ChecksumPatternSpec;

    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    /// Helper function to create test data with valid checksum
    fn create_valid_chunk(chunk: &[u8], checksum_len: usize) -> Vec<u8> {
        let hash = compute_checksum(chunk);
        let mut result = chunk.to_vec();
        result.extend_from_slice(&hash[..checksum_len]);
        result
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_single_match_at_offset_zero(#[case] processor: ProcessorChoice) {
        // Test Case 1: Single valid match at the beginning of data
        let chunk = b"Hello, World!";
        let checksum_len = 4;
        let test_data = create_valid_chunk(chunk, checksum_len);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 1, "Should find exactly one match");
        assert_eq!(matches[0].chunk_start_offset, 0);
        assert_eq!(matches[0].chunk_len, chunk.len());
        assert_eq!(matches[0].checksum_len, checksum_len);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_match_with_offset(#[case] processor: ProcessorChoice) {
        // Test Case 2: Valid match with data before it
        let prefix = b"PADDING_DATA_";
        let chunk = b"Test chunk";
        let checksum_len = 8;

        let mut test_data = prefix.to_vec();
        test_data.extend_from_slice(&create_valid_chunk(chunk, checksum_len));

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 1, "Should find exactly one match");
        assert_eq!(
            matches[0].chunk_start_offset,
            prefix.len() as u64,
            "Match should be at offset after prefix"
        );
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_multiple_matches_different_patterns(#[case] processor: ProcessorChoice) {
        // Test Case 3: Multiple matches with different pattern specs
        let chunk1 = b"First";
        let checksum_len1 = 4;
        let chunk2 = b"Second data block";
        let checksum_len2 = 6;

        let mut test_data = create_valid_chunk(chunk1, checksum_len1);
        test_data.extend_from_slice(b"___SEPARATOR___");
        let offset2 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk2, checksum_len2));

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: chunk1.len(),
                checksum_len: checksum_len1,
            },
            ChecksumPatternSpec {
                chunk_len: chunk2.len(),
                checksum_len: checksum_len2,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 2, "Should find two matches");
        assert_eq!(matches[0].chunk_start_offset, 0);
        assert_eq!(matches[1].chunk_start_offset, offset2 as u64);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_all_zeros_skipped(#[case] processor: ProcessorChoice) {
        // Test Case 4: All-zero data should be skipped
        let chunk_len = 10;
        let checksum_len = 4;
        let test_data = vec![0u8; chunk_len + checksum_len];

        let patterns = vec![ChecksumPatternSpec {
            chunk_len,
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 0, "All-zero patterns should be skipped");
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_invalid_checksum_no_match(#[case] processor: ProcessorChoice) {
        // Test Case 5: Invalid checksum should not match
        let chunk = b"Data with wrong checksum";
        let wrong_checksum = vec![0xFF, 0xFF, 0xFF, 0xFF];

        let mut test_data = chunk.to_vec();
        test_data.extend_from_slice(&wrong_checksum);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len: wrong_checksum.len(),
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(
            matches.len(),
            0,
            "Invalid checksum should not produce matches"
        );
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_insufficient_data_length(#[case] processor: ProcessorChoice) {
        // Test Case 6: Pattern longer than available data
        let chunk = b"Short";
        let checksum_len = 4;
        let test_data = create_valid_chunk(chunk, checksum_len);

        // Request a pattern that's longer than the data
        let patterns = vec![ChecksumPatternSpec {
            chunk_len: test_data.len() + 10,
            checksum_len: 4,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(
            matches.len(),
            0,
            "Should not match when pattern is longer than data"
        );
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_pattern_at_exact_end(#[case] processor: ProcessorChoice) {
        // Test Case 7: Valid pattern ending exactly at data boundary
        let chunk = b"Edge";
        let checksum_len = 3;
        let test_data = create_valid_chunk(chunk, checksum_len);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 1, "Should find match at exact boundary");
        assert_eq!(matches[0].chunk_start_offset, 0);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_minimum_size_chunk(#[case] processor: ProcessorChoice) {
        // Test Case 8: Minimum size chunk (1 byte)
        let chunk = b"X";
        let checksum_len = 2;
        let test_data = create_valid_chunk(chunk, checksum_len);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 1, "Should handle 1-byte chunks");
        assert_eq!(matches[0].chunk_len, 1);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_multiple_patterns_same_offset(#[case] processor: ProcessorChoice) {
        // Test Case 10: Multiple pattern specs that could match at same location
        let chunk = b"Test data here";

        let test_data = create_valid_chunk(chunk, 8);

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: chunk.len(),
                checksum_len: 4,
            },
            ChecksumPatternSpec {
                chunk_len: chunk.len(),
                checksum_len: 8,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        // Both patterns should match at offset 0
        assert_eq!(matches.len(), 2, "Should find matches for both patterns");
        assert_eq!(matches[0].chunk_start_offset, 0);
        assert_eq!(matches[1].chunk_start_offset, 0);
        assert_ne!(
            matches[0].checksum_len, matches[1].checksum_len,
            "Matches should have different checksum lengths"
        );
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_consecutive_valid_chunks(#[case] processor: ProcessorChoice) {
        // Test Case 11: Back-to-back valid chunks
        let chunk1 = b"First";
        let chunk2 = b"Second";
        let checksum_len = 4;

        let mut test_data = create_valid_chunk(chunk1, checksum_len);
        let offset2 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk2, checksum_len));

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: chunk1.len(),
                checksum_len,
            },
            ChecksumPatternSpec {
                chunk_len: chunk2.len(),
                checksum_len,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 2, "Should find both consecutive chunks");
        assert_eq!(matches[0].chunk_start_offset, 0);
        assert_eq!(matches[1].chunk_start_offset, offset2 as u64);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_empty_data(#[case] processor: ProcessorChoice) {
        // Test Case 12: Empty data
        let test_data: Vec<u8> = vec![];
        let patterns = vec![ChecksumPatternSpec {
            chunk_len: 5,
            checksum_len: 4,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 0, "Empty data should produce no matches");
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_empty_patterns(#[case] processor: ProcessorChoice) {
        // Test Case 13: Empty pattern list
        let chunk = b"Some data";
        let test_data = create_valid_chunk(chunk, 4);
        let patterns: Vec<ChecksumPatternSpec> = vec![];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(
            matches.len(),
            0,
            "Empty pattern list should produce no matches"
        );
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_large_checksum_length(#[case] processor: ProcessorChoice) {
        // Test Case 14: Full 32-byte checksum
        let chunk = b"Full checksum";
        let checksum_len = 32; // Full SHA256d output
        let test_data = create_valid_chunk(chunk, checksum_len);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 1, "Should handle full 32-byte checksum");
        assert_eq!(matches[0].checksum_len, 32);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_partial_match_at_end(#[case] processor: ProcessorChoice) {
        // Test Case 15: Incomplete pattern at end of data (should not match)
        let chunk = b"Complete";
        let checksum_len = 4;
        let mut test_data = create_valid_chunk(chunk, checksum_len);

        // Add extra data that could be start of another pattern
        test_data.extend_from_slice(b"Inco");

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        // Should only find the complete pattern, not the partial one
        assert_eq!(matches.len(), 1, "Should not match incomplete patterns");
        assert_eq!(matches[0].chunk_start_offset, 0);
    }

    fn generate_random_array(size: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut data = vec![0u8; size];
        rng.fill_bytes(&mut data);
        data
    }

    /// Test the `generate_random_array()` function, effectively.
    ///
    /// Useful for confirming that the random generation works if the tests using it break.
    #[test]
    fn test_reproducible_random_data() {
        let data = generate_random_array(1024, 42);
        assert_eq!(data[0], 162);

        let data = generate_random_array(1024, 42);
        assert_eq!(data[0], 162);

        // Same seed, longer length.
        let data = generate_random_array(2048, 42);
        assert_eq!(data[0], 162);
    }

    // Optionally mark this as a test to find more seeds with random matches.
    #[allow(dead_code)]
    fn find_random_seeds_with_matches() {
        // Considering patterns 16+4, 20+4, and 32+4, seeds 1-15 have no matches.
        // 16 has a match for 16+4 pattern (now its own test).
        for seed in 17..100000000 {
            println!("Trying seed {}", seed);
            let test_data = generate_random_array(1024 * 1024 * 128, seed);

            let patterns = vec![
                ChecksumPatternSpec::from_str("16+4").unwrap(),
                ChecksumPatternSpec::from_str("20+4").unwrap(),
                ChecksumPatternSpec::from_str("32+4").unwrap(),
            ];

            let matches = search_for_checksums_cpu(&test_data, &patterns);

            assert!(matches.len() == 0, "Match in seed {} - {:?}", seed, matches);
        }
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_with_large_random_data_1(#[case] processor: ProcessorChoice) {
        if !get_test_config().enable_slow_tests {
            return;
        }

        // Test case found by trial-and-error with find_random_seeds_with_matches()
        let test_data = generate_random_array(1024 * 1024 * 128, 16);

        let patterns = vec![
            ChecksumPatternSpec::from_str("16+4").unwrap(),
            ChecksumPatternSpec::from_str("20+4").unwrap(),
            ChecksumPatternSpec::from_str("32+4").unwrap(),
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 2);
        assert_eq!(
            matches[0],
            ChecksumPatternMatch {
                chunk_len: 16,
                checksum_len: 4,
                chunk_start_offset: 62168622,
                chunk_data: vec![
                    245, 173, 89, 145, 217, 220, 26, 82, 193, 168, 246, 37, 103, 48, 19, 220,
                ],
                checksum_data: vec![198, 58, 54, 199],
            }
        );
        assert_eq!(
            matches[1],
            ChecksumPatternMatch {
                chunk_len: 16,
                checksum_len: 4,
                chunk_start_offset: 95768888,
                chunk_data: vec![
                    143, 152, 37, 241, 153, 137, 49, 135, 118, 71, 201, 154, 224, 112, 21, 238,
                ],
                checksum_data: vec![119, 155, 222, 13],
            },
        );
    }
}

// MARK: Adv. Tests

#[cfg(test)]
mod advanced_tests {
    use super::*;
    use rstest::rstest;

    use crate::search_with_cpu::compute_checksum;
    use crate::test_general::get_test_config;
    use crate::types::{ChecksumPatternMatch, ChecksumPatternSpec};

    fn create_valid_chunk(chunk: &[u8], checksum_len: usize) -> Vec<u8> {
        let hash = compute_checksum(chunk);
        let mut result = chunk.to_vec();
        result.extend_from_slice(&hash[..checksum_len]);
        result
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_real_world_bitcoin_style_data(#[case] processor: ProcessorChoice) {
        // Simulate Bitcoin-style transaction with checksum
        let version: u32 = 1;
        let mut tx_data = version.to_le_bytes().to_vec();
        tx_data.push(0x01); // input count
        tx_data.extend_from_slice(b"tx_input_data_here");

        let checksum_len = 4; // Bitcoin uses 4-byte checksums
        let test_data = create_valid_chunk(&tx_data, checksum_len);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: tx_data.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].chunk_start_offset, 0);
        assert_eq!(matches[0].checksum_len, 4);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_performance_no_matches_large_data(#[case] processor: ProcessorChoice) {
        // Test with 1MB of random-ish data (no valid checksums).
        let test_data: Vec<u8> = (0..get_test_config().max_data_size_mebibytes)
            .map(|i| ((i * 1103515245 + 12345) % 256) as u8)
            .collect();

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: 32,
                checksum_len: 4,
            },
            ChecksumPatternSpec {
                chunk_len: 64,
                checksum_len: 8,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        // Very unlikely to have any matches in random data.
        assert!(matches.len() == 0);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_multiple_checksums_different_lengths(#[case] processor: ProcessorChoice) {
        // Test finding patterns with varying checksum lengths
        // Note: When a longer checksum is present, shorter checksum patterns will also match
        // because they check a prefix of the same hash
        let chunk = b"Same chunk data";

        let mut test_data = Vec::new();

        // First pattern: chunk + 2-byte checksum
        let offset_2 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk, 2));
        test_data.extend_from_slice(b"___"); // separator

        // Second pattern: chunk + 4-byte checksum
        // This will match BOTH 2-byte and 4-byte patterns (first 2 bytes are valid too)
        let offset_4 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk, 4));
        test_data.extend_from_slice(b"___"); // separator

        // Third pattern: chunk + 8-byte checksum
        // This will match ALL patterns (2, 4, and 8 byte)
        let offset_8 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk, 8));

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: chunk.len(),
                checksum_len: 2,
            },
            ChecksumPatternSpec {
                chunk_len: chunk.len(),
                checksum_len: 4,
            },
            ChecksumPatternSpec {
                chunk_len: chunk.len(),
                checksum_len: 8,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        // Expected matches:
        // - offset_2: 2-byte pattern (1 match)
        // - offset_4: 2-byte and 4-byte patterns (2 matches)
        // - offset_8: 2-byte, 4-byte, and 8-byte patterns (3 matches)
        // Total: 6 matches
        assert_eq!(matches.len(), 6, "Should find 6 total matches (1+2+3)");

        // Count matches by offset and checksum length
        let matches_at_offset_2: Vec<_> = matches
            .iter()
            .filter(|m| m.chunk_start_offset == offset_2 as u64)
            .collect();
        let matches_at_offset_4: Vec<_> = matches
            .iter()
            .filter(|m| m.chunk_start_offset == offset_4 as u64)
            .collect();
        let matches_at_offset_8: Vec<_> = matches
            .iter()
            .filter(|m| m.chunk_start_offset == offset_8 as u64)
            .collect();

        assert_eq!(
            matches_at_offset_2.len(),
            1,
            "Should find 1 match at offset_2"
        );
        assert_eq!(
            matches_at_offset_4.len(),
            2,
            "Should find 2 matches at offset_4"
        );
        assert_eq!(
            matches_at_offset_8.len(),
            3,
            "Should find 3 matches at offset_8"
        );

        // Verify specific matches at offset_2
        assert!(matches_at_offset_2.iter().any(|m| m.checksum_len == 2));

        // Verify specific matches at offset_4
        assert!(matches_at_offset_4.iter().any(|m| m.checksum_len == 2));
        assert!(matches_at_offset_4.iter().any(|m| m.checksum_len == 4));

        // Verify specific matches at offset_8
        assert!(matches_at_offset_8.iter().any(|m| m.checksum_len == 2));
        assert!(matches_at_offset_8.iter().any(|m| m.checksum_len == 4));
        assert!(matches_at_offset_8.iter().any(|m| m.checksum_len == 8));
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_non_overlapping_checksum_patterns(#[case] processor: ProcessorChoice) {
        // Alternative test: use different chunk data to avoid overlapping matches.
        let chunk1 = b"First chunk";
        let chunk2 = b"Second chunk";
        let chunk3 = b"Third chunk";

        let mut test_data = Vec::new();

        let offset_1 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk1, 4));
        test_data.extend_from_slice(b"___");

        let offset_2 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk2, 4));
        test_data.extend_from_slice(b"___");

        let offset_3 = test_data.len();
        test_data.extend_from_slice(&create_valid_chunk(chunk3, 8));

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: chunk1.len(),
                checksum_len: 4,
            },
            ChecksumPatternSpec {
                chunk_len: chunk2.len(),
                checksum_len: 4,
            },
            ChecksumPatternSpec {
                chunk_len: chunk3.len(),
                checksum_len: 8,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        // Should find exactly 4 matches, one for each pattern, plus the time chunk3 + 8 bytes is found
        // with the 4-byte checksum.
        assert_eq!(
            matches.len(),
            4,
            "Should find exactly 4 matches. Found: {:?}",
            matches
        );

        // Verify each match.
        assert_eq!(
            matches[0],
            ChecksumPatternMatch {
                chunk_len: chunk1.len(),
                checksum_len: 4,
                chunk_start_offset: offset_1 as u64,
                chunk_data: chunk1.to_vec(),
                checksum_data: vec![121, 164, 27, 175],
            }
        );
        assert_eq!(
            matches[1],
            ChecksumPatternMatch {
                chunk_len: chunk2.len(),
                checksum_len: 4,
                chunk_start_offset: offset_2 as u64,
                chunk_data: chunk2.to_vec(),
                checksum_data: vec![59, 20, 82, 250],
            }
        );

        // Weird one:
        assert_eq!(chunk1.len(), chunk3.len());
        assert_eq!(
            matches[2],
            ChecksumPatternMatch {
                chunk_len: chunk3.len(),
                checksum_len: 4,
                chunk_start_offset: offset_3 as u64,
                chunk_data: chunk3.to_vec(),
                checksum_data: vec![229, 9, 220, 119],
            }
        );

        // Nominal result.
        assert_eq!(
            matches[3],
            ChecksumPatternMatch {
                chunk_len: chunk3.len(),
                checksum_len: 8,
                chunk_start_offset: offset_3 as u64,
                chunk_data: chunk3.to_vec(),
                checksum_data: vec![229, 9, 220, 119, 17, 114, 177, 90], // Superstring of case above.
            }
        );
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_checksum_collision_detection(#[case] processor: ProcessorChoice) {
        // Test that only exact checksum matches are found
        let chunk = b"Original data";
        let checksum_len = 4;

        let correct_data = create_valid_chunk(chunk, checksum_len);
        let correct_checksum = &correct_data[chunk.len()..];

        // Create similar but incorrect checksum (flip one bit)
        let mut incorrect_data = chunk.to_vec();
        let mut wrong_checksum = correct_checksum.to_vec();
        wrong_checksum[0] ^= 0x01; // Flip one bit
        incorrect_data.extend_from_slice(&wrong_checksum);

        let mut test_data = correct_data.clone();
        test_data.extend_from_slice(b"___SEP___");
        test_data.extend_from_slice(&incorrect_data);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(
            matches.len(),
            1,
            "Should only match exact checksum, not near-miss"
        );
        assert_eq!(matches[0].chunk_start_offset, 0);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_nested_valid_patterns(#[case] processor: ProcessorChoice) {
        // Create a pattern where one valid chunk+checksum contains another
        let inner_chunk = b"Inner";
        let inner_checksum_len = 4;
        let inner_data = create_valid_chunk(inner_chunk, inner_checksum_len);

        // Use the inner data as part of outer chunk
        let mut outer_chunk = b"Prefix_".to_vec();
        outer_chunk.extend_from_slice(&inner_data);
        outer_chunk.extend_from_slice(b"_Suffix");

        let outer_checksum_len = 6;
        let test_data = create_valid_chunk(&outer_chunk, outer_checksum_len);

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: inner_chunk.len(),
                checksum_len: inner_checksum_len,
            },
            ChecksumPatternSpec {
                chunk_len: outer_chunk.len(),
                checksum_len: outer_checksum_len,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        // Should find both the inner and outer patterns
        assert!(matches.len() >= 2, "Should find nested patterns");

        // Verify we found both patterns
        let has_inner = matches
            .iter()
            .any(|m| m.chunk_len == inner_chunk.len() && m.checksum_len == inner_checksum_len);
        let has_outer = matches
            .iter()
            .any(|m| m.chunk_len == outer_chunk.len() && m.checksum_len == outer_checksum_len);

        assert!(has_inner, "Should find inner pattern");
        assert!(has_outer, "Should find outer pattern");
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_boundary_offset_calculation(#[case] processor: ProcessorChoice) {
        // Verify offset calculations are correct at various positions
        let chunk = b"PATTERN";
        let checksum_len = 4;
        let pattern_data = create_valid_chunk(chunk, checksum_len);

        let test_positions = vec![0, 100, 1000, 10000];

        for pos in test_positions {
            let mut test_data = vec![0xAA; pos]; // Fill with 0xAA
            let expected_offset = test_data.len();
            test_data.extend_from_slice(&pattern_data);
            test_data.extend_from_slice(&vec![0xBB; 100]); // Trailing data

            let patterns = vec![ChecksumPatternSpec {
                chunk_len: chunk.len(),
                checksum_len,
            }];

            let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

            assert!(
                matches.len() >= 1,
                "Should find pattern at position {}",
                pos
            );
            assert_eq!(
                matches
                    .iter()
                    .find(|m| m.chunk_len == chunk.len())
                    .unwrap()
                    .chunk_start_offset,
                expected_offset as u64,
                "Offset calculation incorrect at position {}",
                pos
            );
        }
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_single_byte_variations(#[case] processor: ProcessorChoice) {
        // Test that changing a single byte in chunk breaks the match
        let chunk = b"Test data for verification";
        let checksum_len = 8;
        let test_data = create_valid_chunk(chunk, checksum_len);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        // Original should match
        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();
        assert_eq!(matches.len(), 1);

        // Modify one byte in chunk - should not match
        let mut modified_data = test_data.clone();
        modified_data[5] ^= 0xFF;
        let matches_modified = search_for_checksums_cpu(&modified_data, &patterns);
        assert_eq!(matches_modified.len(), 0, "Modified chunk should not match");

        // Modify one byte in checksum - should not match
        let mut modified_checksum = test_data.clone();
        modified_checksum[chunk.len() + 2] ^= 0xFF;
        let matches_checksum = search_for_checksums_cpu(&modified_checksum, &patterns);
        assert_eq!(
            matches_checksum.len(),
            0,
            "Modified checksum should not match"
        );
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_all_zeros_with_valid_checksum_nearby(#[case] processor: ProcessorChoice) {
        // Ensure all-zeros are skipped even when valid patterns exist nearby
        let zeros = vec![0u8; 20];
        let chunk = b"Valid";
        let checksum_len = 4;

        let mut test_data = zeros.clone();
        test_data.extend_from_slice(&create_valid_chunk(chunk, checksum_len));

        let patterns = vec![
            ChecksumPatternSpec {
                chunk_len: 10,
                checksum_len: 4,
            },
            ChecksumPatternSpec {
                chunk_len: chunk.len(),
                checksum_len,
            },
        ];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        // Should only find the valid chunk, not the zeros
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].chunk_start_offset, zeros.len() as u64);
    }

    #[rstest]
    #[case(ProcessorChoice::Cpu)]
    #[case(ProcessorChoice::Gpu)]
    fn test_maximum_checksum_length(#[case] processor: ProcessorChoice) {
        // Test with maximum possible checksum (full SHA256d output)
        let chunk = b"Max checksum test";
        let checksum_len = 32;
        let test_data = create_valid_chunk(chunk, checksum_len);

        let patterns = vec![ChecksumPatternSpec {
            chunk_len: chunk.len(),
            checksum_len,
        }];

        let matches = search_for_checksums(&test_data, &patterns, processor).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].checksum_len, 32);

        // Verify the full checksum matches
        let expected_hash = compute_checksum(chunk);
        let actual_checksum = &test_data[chunk.len()..chunk.len() + 32];
        assert_eq!(actual_checksum, &expected_hash[..32]);
    }
}
