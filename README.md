# armory-wallet-checksum-searcher

A tool to locate and extract Armory wallets on corrupted and deleted hard drives.

## Features

* Searches a drive image for deleted and corrupted armory wallet files, based on a checksum property within the file.
* Writes to a log file and ndjson.
* Supports raw disk images (`dd`-style).

### Benchmarks

* 8 MiB/s - 1 TiB takes 1.5 days

## Process

* Armory wallets have a 32-byte private key, followed by a 4-byte sha256d hash of that key.
* By taking every group of 36 bytes on the drive image, performing the checksum validation (a sha256d hash) on the first 32 bytes, and seeing if they match the final 4 bytes, we can find all parts that are "probably private keys".
* After you find these keys, use a tool like `ku` to convert them to usable keys.

## Usage

Install from crates.io, and then run:

```bash
cargo install armory_wallet_checksum_searcher
armory_wallet_checksum_searcher -i input_file.img --log ./output_log.log --ndjson ./output_records.ndjson
```

Or, clone from source and run:

```bash
git clone https://github.com/RecRanger/armory-wallet-checksum-searcher
cargo run --release -- -i input_file.img --log ./output_log.log --ndjson ./output_records.ndjson
sudo cargo run --release -- -i /dev/sda --log ./output_log.log --ndjson ./output_records.ndjson
```

## Contributing

Please Star this repo if it's helpful. Open Issues.
