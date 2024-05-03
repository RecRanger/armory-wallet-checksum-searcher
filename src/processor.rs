
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::collections::VecDeque;
use std::time::{Instant, Duration};

use sha2::{Sha256, Digest};
use lz4_flex::frame::FrameDecoder;
use std::path::PathBuf;
use indexmap::IndexMap;

use log::{info, warn, error};

use crate::types::{ChecksumPatternSpec, ChecksumPatternMatch};


fn sha256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

fn hash256(data: &[u8]) -> Vec<u8> {
    sha256(&sha256(data))
}

fn compute_checksum(data: &[u8], len: usize) -> Vec<u8> {
    hash256(data)[..len].to_vec()
}


fn search_block_for_checksum(block: &[u8], block_start_idx: i64, checksum_patterns: &[ChecksumPatternSpec]) -> Vec<ChecksumPatternMatch> {
    let mut matches: Vec<ChecksumPatternMatch> = Vec::new();
    
    for i in 0..block.len() {
        for ChecksumPatternSpec { chunk_len, checksum_len } in checksum_patterns {
            if i + chunk_len + checksum_len > block.len() {
                // We're at the end, and it's too long to match
                // This method wastes a few loop cycles at the end, but it's easier than setting the upper bound of the loop smartly.
                continue;
            }
            let chunk = &block[i..i + chunk_len];
            let checksum = &block[i + chunk_len..i + chunk_len + checksum_len];

            // if chunk is all zeros, skip
            if chunk.iter().all(|&x| x == 0) {
                continue;
            }

            // if checksum is all zeros, skip
            if checksum.iter().all(|&x| x == 0) {
                continue;
            }

            let hash_result = compute_checksum(chunk, *checksum_len);

            if hash_result == checksum {
                let match_offset = block_start_idx + (i as i64);
                info!("✅ Match! Offset in File: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}, Hash: {:x?}",
                    match_offset, match_offset,
                    chunk_len,
                    chunk,
                    hash_result
                );

                matches.push(ChecksumPatternMatch {
                    checksum_pattern: ChecksumPatternSpec { chunk_len: *chunk_len, checksum_len: *checksum_len },
                    offset: match_offset,
                });
            }
        }
    }
    
    matches
}


pub fn process_file(file_path: &PathBuf, block_size: usize, checksum_patterns: &[ChecksumPatternSpec]) -> io::Result<()> {
    let file = File::open(file_path)?;
    let mut reader: Box<dyn Read> = if file_path.extension().and_then(std::ffi::OsStr::to_str) == Some("lz4") {
        info!("Detected LZ4 file. Using LZ4 frame decoder.");
        Box::new(FrameDecoder::new(file))
    } else {
        info!("Using BufReader to read from regular binary file (i.e., assuming no compression).");
        Box::new(BufReader::new(file))
    };

    let longest_chunk_and_checksum_bytes: usize = checksum_patterns.iter().map(|q| q.chunk_len + q.checksum_len).max().unwrap();
    info!("Longest chunk and checksum bytes: {}", longest_chunk_and_checksum_bytes);

    let overlap_length = longest_chunk_and_checksum_bytes; // maybe this should be "-1", but doesn't matter
    let mut block = vec![0u8; block_size + overlap_length];
    let mut overlap = vec![0u8; overlap_length];
    let start_time = Instant::now();
    let mut total_success_count = 0;
    let mut initial_read = true;

    let mut block_start_idx: i64 = -(overlap_length as i64);
    let mut total_bytes_read = 0;
    let mut last_log_time = Instant::now();
    let mut recent_speeds: VecDeque<(Instant, f32)> = VecDeque::new();

    // Populating the IndexMap with 0-counts at start.
    // Using an IndexMap instead of a HashMap to keep the order of insertion.
    let mut checksum_pattern_success_count: IndexMap<String, i32> = IndexMap::new();
    for checksum_pattern in checksum_patterns {
        checksum_pattern_success_count.insert(checksum_pattern.to_string(), 0);
    }

    // Loop through each "block" of data from the file
    loop {
        let had_success_this_block;
        let start_time_this_block = Instant::now();

        // Prepend overlap if not the first read
        if !initial_read {
            block[..overlap_length].copy_from_slice(&overlap);
        }

        // Read data into the block after the overlap
        match reader.read(&mut block[overlap_length..]) {
            Ok(0) => {
                info!("No data was read this time, meaning we reached the end of the file. Breaking!");
                break // Break if no data was read
            }
            Ok(n) => {
                let read_size = overlap_length + n; // Total data size including overlap
                total_bytes_read += read_size;
                
                // DO THE SEARCH:
                let pattern_matches = search_block_for_checksum(
                    &block[..read_size], block_start_idx, checksum_patterns);

                // Update success counts
                had_success_this_block = pattern_matches.len() > 0;
                total_success_count += pattern_matches.len();
                for pattern_match in pattern_matches {
                    let pattern_str = pattern_match.checksum_pattern.to_string();
                    let count = checksum_pattern_success_count.entry(pattern_str).or_insert(0);
                    *count += 1;
                }

                // Update block_start_idx for the next iteration
                block_start_idx += n as i64;

                // Update overlap for the next iteration
                overlap.copy_from_slice(&block[read_size - overlap_length..read_size]);

                // Update recent speeds
                let this_block_speed_bytes_per_sec = read_size as f32 / start_time_this_block.elapsed().as_secs_f32();
                recent_speeds.push_back((Instant::now(), this_block_speed_bytes_per_sec));  
            }
            Err(e) => {
                error!("Error reading data: {:?}", e);
                return Err(e)
            }
        }

        initial_read = false;

        // log progress
        if last_log_time.elapsed() > Duration::from_secs(45) || had_success_this_block {
            let time_elapsed = start_time.elapsed().as_secs_f32();
            let bytes_per_sec = total_bytes_read as f32 / time_elapsed;

            // Remove old entries that are outside the 60-second window, then compute the average speed
            while recent_speeds.front().map_or(false, |&(time, _)| Instant::now().duration_since(time) > Duration::from_secs(60)) {
                recent_speeds.pop_front();
            }
            let average_speed_bytes_per_sec = if recent_speeds.is_empty() {
                warn!("No recent speeds to compute SMA. Using 0.0 KiB/sec. Seems like the process is stuck/halted.");
                0.0
            } else {
                recent_speeds.iter().map(|&(_, speed)| speed).sum::<f32>() / recent_speeds.len() as f32
            };

            info!("PROGRESS: {} bytes = {} MiB = {} GiB read @ {:.1}|{:.1} KiB/sec (all|60sec). Successes: {} = {:?}",
                total_bytes_read,
                (total_bytes_read as f32 / 1024.0 / 1024.0 * 10.0).round() / 10.0,
                (total_bytes_read as f32 / 1024.0 / 1024.0 / 1024.0 * 10.0).round() / 10.0,
                bytes_per_sec / 1024.0,
                average_speed_bytes_per_sec / 1024.0,
                total_success_count,
                checksum_pattern_success_count,
            );
            last_log_time = Instant::now();
        }
    }

    let duration = start_time.elapsed();
    info!("Completed in {:?}, Total successes: {}", duration, total_success_count);

    Ok(())
}

