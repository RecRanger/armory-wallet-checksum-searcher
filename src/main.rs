mod processor;
mod types;

use processor::process_file;
use types::ChecksumPatternSpec;

use std::io::{self};
use std::time::SystemTime;

use clap::Parser;
use std::path::PathBuf;

use fern::Dispatch;
use log::{info, error};



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


#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    #[clap(short, long, value_parser)]
    file: PathBuf,

    #[clap(short, long, value_parser)]
    output: PathBuf,

    // block size in KiB, alias --block-size
    #[clap(short, long="block-size", default_value = "10240")]
    block_size_kibibytes: usize,
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
    let block_size_bytes = args.block_size_kibibytes * 1024;

    let checksum_patterns = vec![
        ChecksumPatternSpec { chunk_len: 16, checksum_len: 4 }, // Initialization Vector (IV)
        ChecksumPatternSpec { chunk_len: 20, checksum_len: 4 }, // Public Key Hash160 (Address)
        ChecksumPatternSpec { chunk_len: 32, checksum_len: 4 }, // ChainCode and PrivKey
        ChecksumPatternSpec { chunk_len: 44, checksum_len: 4 }, // KdfParameters: function width of the KDF parameters block
        ChecksumPatternSpec { chunk_len: 65, checksum_len: 4 }, // "Public Key"
        ChecksumPatternSpec { chunk_len: 38, checksum_len: 4 }, // an arbitrary-length searcher to act as a "control"
    ];
    info!("Using checksum patterns: {:?}", checksum_patterns);
    
    info!("Using block size: {} bytes = {} MiB", block_size_bytes, (block_size_bytes as f32 / 1024.0 / 1024.0));
    process_file(&args.file, block_size_bytes, &checksum_patterns)?;

    Ok(())
}
