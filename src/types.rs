use std::str::FromStr;
use serde::Serialize;


#[derive(Debug, Clone)]
pub struct ChecksumPatternSpec {
    pub chunk_len: usize,
    pub checksum_len: usize,
    // TODO: Could add hash function field here, if we wanted.
}

impl ToString for ChecksumPatternSpec {
    fn to_string(&self) -> String {
        format!("{}+{}", self.chunk_len, self.checksum_len)
    }
}

impl FromStr for ChecksumPatternSpec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('+').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid checksum pattern: {}", s));
        }

        let chunk_len = parts[0]
            .parse::<usize>()
            .map_err(|e| format!("Invalid chunk length: {}", e))?;
        let checksum_len = parts[1]
            .parse::<usize>()
            .map_err(|e| format!("Invalid checksum length: {}", e))?;

        Ok(ChecksumPatternSpec {
            chunk_len,
            checksum_len,
        })
    }
}

impl ChecksumPatternSpec {
    pub fn total_length(&self) -> usize {
        self.chunk_len + self.checksum_len
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChecksumPatternMatch {
    pub chunk_len: usize,
    pub checksum_len: usize,

    pub chunk_start_offset: u64,

    pub chunk_data: Vec<u8>,
    pub checksum_data: Vec<u8>,
}

impl ChecksumPatternMatch {
    pub fn to_checksum_pattern_string(&self) -> String {
        format!("{}+{}", self.chunk_len, self.checksum_len)
    }
}
