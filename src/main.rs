mod processor;
mod processor_gpu;
mod types;

use processor::process_file;
use types::ChecksumPatternSpec;

use std::fs::File;
use std::io::{self, BufWriter, Write as _};
use std::time::SystemTime;

use clap::Parser;
use std::path::PathBuf;

use fern::Dispatch;
use log::{error, info};

fn setup_logger(log_file: Option<PathBuf>) -> Result<(), fern::InitError> {
    let mut dispatch = Dispatch::new()
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
        .chain(std::io::stdout());

    if let Some(log_file_path) = log_file {
        dispatch = dispatch.chain(fern::log_file(log_file_path)?);
    }

    dispatch.apply()?;
    Ok(())
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Input file path (generally .img or .tar file)
    #[clap(short, long, value_parser)]
    input_file: PathBuf,

    /// Output log file path
    #[clap(short = 'l', long, alias = "log", value_parser)]
    output_log_file: Option<PathBuf>,

    /// Output ndjson file path
    #[clap(short, long, alias = "ndjson", value_parser)]
    output_ndjson_file: Option<PathBuf>,

    /// Checksum patterns to search for (chunk length + checksum length).
    /// Use `--help` to see full details.
    ///
    /// Specify multiple, like: "-p 16+4 -p 20+4" etc.
    ///
    /// Default: "-p 16+4 -p 20+4 -p 32+4 -p 44+4 -p 65+4 -p 38+4".
    ///
    /// 16+4 -> Initialization Vector (IV)
    ///
    /// 20+4 -> Public Key Hash160 (Address)
    ///
    /// 32+4 -> ChainCode and PrivKey
    ///
    /// 44+4 -> KdfParameters: function width of the KDF parameters block
    ///
    /// 65+4 -> Public Key
    ///
    /// 38+4 -> An arbitrary-length searcher to act as a "control"
    #[clap(short, long = "pattern", value_parser)]
    patterns: Vec<ChecksumPatternSpec>,
}

impl Args {
    fn parse_with_defaults() -> Self {
        let mut args = Args::parse();
        if args.patterns.is_empty() {
            args.patterns = Args::default_patterns();
        }
        args
    }

    fn default_patterns() -> Vec<ChecksumPatternSpec> {
        vec!["16+4", "20+4", "32+4", "44+4", "65+4", "38+4"]
            .iter()
            .map(|&s| s.parse().unwrap())
            .collect()
    }
}

fn main() -> io::Result<()> {
    let args = Args::parse_with_defaults();

    // Register log file and/or normal stdout logger.
    match setup_logger(args.output_log_file) {
        Ok(_) => info!("Logger setup successful!"),
        Err(e) => {
            error!("Error setting up logger: {:?}", e);
            panic!("Error setting up logger: {:?}", e);
        }
    }

    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    info!("Starting processing of file: {:?}", args.input_file);

    let checksum_patterns = &args.patterns;
    info!(
        "Searching for {} checksum patterns: {:?}",
        checksum_patterns.len(),
        checksum_patterns
    );
    let pattern_matches = process_file(&args.input_file, &checksum_patterns)?;
    info!("Found {} pattern matches.", pattern_matches.len());

    if let Some(ref output_ndjson_path) = args.output_ndjson_file {
        let output_file = File::create(&output_ndjson_path)?;
        let mut output_file_writer = BufWriter::new(output_file);

        for pattern_match in &pattern_matches {
            let json = serde_json::to_string(&pattern_match)?;
            writeln!(output_file_writer, "{}", json)?;
        }
    }

    Ok(())
}
