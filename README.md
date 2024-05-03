# armory-wallet-checksum-searcher

A tool to locate and extract Armory wallets on corrupted and deleted hard drives.

## Features

* Searches a drive image for deleted and corrupted armory wallet files, based on a checksum property within the file.
* Writes to a log file.
* Supports both raw images (`dd`-style) and lz4-compressed images (`.img.lz4`).
    * Open a feature request if you want support for more formats.

### Benchmarks

* Rust (in `--release` mode): 8 MiB/s - 1 TiB takes 1.5 days.

## Process

* Armory wallets have a 32-byte private key, followed by a 4-byte sha256d hash of that key.
* By taking every group of 36 bytes on the drive image, performing the checksum validation (a sha256d hash) on the first 32 bytes, and seeing if they match the final 4 bytes, we can find all parts that are "probably private keys".
* After you find these keys, use a tool like `ku` to convert them to usable keys. I'm not at that point yet.


## Usage

Install from crates.io, and then run:

```bash
cargo install armory_wallet_checksum_searcher
armory_wallet_checksum_searcher -f input_file.img -o ./output_log.log
```

Or, clone from source and run:

```bash
git clone https://github.com/RecRanger/armory-wallet-checksum-searcher
cargo run --release -- -f input_file.img -o ./output_log.log
sudo cargo run --release -- -f /dev/sda -o ./output_log.log
```

## Contributing

Please Star this repo if it's helpful. Open Issues.
