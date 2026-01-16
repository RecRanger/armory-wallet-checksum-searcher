use std::str::FromStr;
use serde::Serialize;


#[derive(Debug, Clone)]
pub struct ChecksumPatternSpec {
    pub chunk_len: u16,
    pub checksum_len: u16,
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
            .parse::<u16>()
            .map_err(|e| format!("Invalid chunk length: {}", e))?;
        let checksum_len = parts[1]
            .parse::<u16>()
            .map_err(|e| format!("Invalid checksum length: {}", e))?;

        Ok(ChecksumPatternSpec {
            chunk_len,
            checksum_len,
        })
    }
}

impl ChecksumPatternSpec {
    pub fn total_length(&self) -> u16 {
        self.chunk_len + self.checksum_len
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChecksumPatternMatch {
    pub chunk_len: u16,
    pub checksum_len: u16,

    pub chunk_start_offset: u64,
}

impl ChecksumPatternMatch {
    pub fn to_checksum_pattern_string(&self) -> String {
        format!("{}+{}", self.chunk_len, self.checksum_len)
    }
}
