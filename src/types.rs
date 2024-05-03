use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct ChecksumPatternSpec {
    pub chunk_len: usize,
    pub checksum_len: usize,
    // TODO: could add hash function field here, if we wanted
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

        let chunk_len = parts[0].parse::<usize>().map_err(|e| format!("Invalid chunk length: {}", e))?;
        let checksum_len = parts[1].parse::<usize>().map_err(|e| format!("Invalid checksum length: {}", e))?;

        Ok(ChecksumPatternSpec { chunk_len, checksum_len })
    }
}



pub struct ChecksumPatternMatch {
    pub checksum_pattern: ChecksumPatternSpec,

    #[allow(dead_code)]
    pub offset: i64,
}
