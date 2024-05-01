use std::fs::File;
use std::io::{self, BufReader, Read};
use std::time::{Instant, Duration};
use sha2::{Sha256, Digest};
use lz4_flex::frame::FrameDecoder;
use clap::Parser;
use std::path::PathBuf;

use fern::Dispatch;
use log::{info, error};
use std::time::SystemTime;


#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    #[clap(short, long, value_parser)]
    file: PathBuf,

    #[clap(short, long, value_parser)]
    output: PathBuf,
}

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


fn search_block_for_checksum(block: &[u8], block_start_idx: i64, search_queries: &[SearchQuery]) -> usize {
    let mut success_count = 0;
    
    for i in 0..block.len() {
        for SearchQuery { chunk_len, checksum_len } in search_queries {
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
                info!("✅ Match! Offset in File: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}, Hash: {:x?}",
                    block_start_idx + (i as i64), block_start_idx + (i as i64),
                    chunk_len,
                    chunk,
                    hash_result
                );

                success_count += 1;
            }
        }
    }
    
    success_count
}


fn process_file(file_path: &PathBuf, block_size: usize, search_queries: &[SearchQuery]) -> io::Result<()> {
    let file = File::open(file_path)?;
    let mut reader: Box<dyn Read> = if file_path.extension().and_then(std::ffi::OsStr::to_str) == Some("lz4") {
        info!("Detected LZ4 file. Using LZ4 frame decoder.");
        Box::new(FrameDecoder::new(file))
    } else {
        info!("Using BufReader to read from regular binary file (i.e., assuming no compression).");
        Box::new(BufReader::new(file))
    };

    let longest_chunk_and_checksum_bytes: usize = search_queries.iter().map(|q| q.chunk_len + q.checksum_len).max().unwrap();
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

    loop {
        let had_success_this_time;

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
                let success_count_this_time = search_block_for_checksum(
                    &block[..read_size], block_start_idx, search_queries);

                had_success_this_time = success_count_this_time > 0;
                total_success_count += success_count_this_time;
                block_start_idx += n as i64;

                // Update overlap for the next iteration
                overlap.copy_from_slice(&block[read_size - overlap_length..read_size]);
            }
            Err(e) => {
                error!("Error reading data: {:?}", e);
                return Err(e)
            }
        }

        initial_read = false;

        // log progress
        if last_log_time.elapsed() > Duration::from_secs(45) || had_success_this_time {
            let time_elapsed = start_time.elapsed().as_secs_f32();
            let bytes_per_sec = total_bytes_read as f32 / time_elapsed;

            info!("PROGRESS: {} bytes = {} MiB read @ {:.2} KiB/sec. Total successes: {}. ",
                total_bytes_read,
                (total_bytes_read as f32 / 1024.0 / 1024.0).round(),
                bytes_per_sec / 1024.0,
                total_success_count);
            last_log_time = Instant::now();
        }
    }

    let duration = start_time.elapsed();
    info!("Completed in {:?}, Total successes: {}", duration, total_success_count);

    Ok(())
}


fn setup_logger(log_file: &PathBuf) -> Result<(), fern::InitError> {
    Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                humantime::format_rfc3339_seconds(SystemTime::now()),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(log::LevelFilter::Debug)
        .chain(std::io::stdout())
        .chain(fern::log_file(log_file)?)
        .apply()?;
    Ok(())
}

#[derive(Debug)]
struct SearchQuery {
    chunk_len: usize,
    checksum_len: usize,
    // TODO: could add hash function here, if we wanted
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    match setup_logger(&args.output) {
        Ok(_) => info!("Logger setup successful!"),
        Err(e) => {
            error!("Error setting up logger: {:?}", e);
            panic!("Error setting up logger: {:?}", e);
        }
    }

    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    info!("Starting processing of file: {:?}", args.file);
    let block_size_bytes = 10 * 1024 * 1024;

    let search_queries = vec![
        SearchQuery { chunk_len: 16, checksum_len: 4 }, // Initialization Vector (IV)
        SearchQuery { chunk_len: 20, checksum_len: 4 }, // Public Key Hash160 (Address)
        SearchQuery { chunk_len: 32, checksum_len: 4 }, // ChainCode and PrivKey
        SearchQuery { chunk_len: 44, checksum_len: 4 }, // KdfParameters: function width of the KDF parameters block
        SearchQuery { chunk_len: 65, checksum_len: 4 }, // "Public Key"
    ];
    info!("Using search queries: {:?}", search_queries);
    
    info!("Using block size: {} bytes = {} MiB", block_size_bytes, (block_size_bytes as f32 / 1024.0 / 1024.0).round());
    process_file(&args.file, block_size_bytes, &search_queries)?;

    Ok(())
}
