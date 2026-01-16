use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use sha2::{Digest, Sha256};

use log::info;

use crate::types::{ChecksumPatternMatch, ChecksumPatternSpec};

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn sha256d(data: &[u8]) -> [u8; 32] {
    sha256(&sha256(data))
}

fn compute_checksum(data: &[u8]) -> [u8; 32] {
    sha256d(data)
}

fn search_for_checksums(
    data: &[u8],
    checksum_patterns: &[ChecksumPatternSpec],
) -> Vec<ChecksumPatternMatch> {
    let mut matches: Vec<ChecksumPatternMatch> = Vec::new();

    // Create progress bar.
    let progress_bar = ProgressBar::new(data.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({binary_bytes_per_sec}) - ETA {eta}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for chunk_start_idx in 0..data.len() {
        // Update progress bar every 1MB to avoid overhead.
        if chunk_start_idx % 1_000_000 == 0 {
            progress_bar.set_position(chunk_start_idx as u64);
        }

        for pattern in checksum_patterns {
            if chunk_start_idx + pattern.total_length() as usize > data.len() {
                continue;
            }

            let chunk = &data[chunk_start_idx..chunk_start_idx + (pattern.chunk_len as usize)];
            let checksum = &data[chunk_start_idx + (pattern.chunk_len as usize)
                ..chunk_start_idx + pattern.total_length() as usize];

            // Skip if chunk is all zeros.
            if chunk.iter().all(|&x| x == 0) {
                continue;
            }

            // Skip if checksum is all zeros.
            if checksum.iter().all(|&x| x == 0) {
                continue;
            }

            let hash_result = compute_checksum(chunk);

            if hash_result == checksum {
                info!(
                    "✅ Match! Offset: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}, Hash: {:x?}",
                    chunk_start_idx, chunk_start_idx, pattern.chunk_len, chunk, hash_result
                );

                matches.push(ChecksumPatternMatch {
                    chunk_len: pattern.chunk_len,
                    checksum_len: pattern.checksum_len,
                    chunk_start_offset: chunk_start_idx as u64,
                });
            }
        }
    }

    progress_bar.finish_with_message("Search complete");

    matches
}

pub fn process_file(
    file_path: &PathBuf,
    checksum_patterns: &[ChecksumPatternSpec],
) -> io::Result<Vec<ChecksumPatternMatch>> {
    let file = File::open(file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    info!(
        "Memory-mapped file: {} bytes ({:.3} GiB)",
        mmap.len(),
        (mmap.len() as f64 / 1024.0 / 1024.0 / 1024.0)
    );

    let start_time = Instant::now();

    // Initialize success count map
    let mut checksum_pattern_success_count: IndexMap<String, i32> = IndexMap::new();
    for checksum_pattern in checksum_patterns {
        checksum_pattern_success_count.insert(checksum_pattern.to_string(), 0);
    }

    // Search the entire file
    let pattern_matches = search_for_checksums(&mmap[..], checksum_patterns);

    // Update success counts
    let total_success_count = pattern_matches.len();
    for pattern_match in &pattern_matches {
        let pattern_str = pattern_match.to_checksum_pattern_string();
        let count = checksum_pattern_success_count
            .entry(pattern_str)
            .or_insert(0);
        *count += 1;
    }

    let duration = start_time.elapsed();
    info!(
        "Completed in {:?}, Total successes: {} = {:?}",
        duration, total_success_count, checksum_pattern_success_count
    );

    Ok(pattern_matches)
}
