

#[derive(Debug)]
pub struct ChecksumPatternSpec {
    pub chunk_len: usize,
    pub checksum_len: usize,
    // TODO: could add hash function field here, if we wanted
}

impl ChecksumPatternSpec {
    pub fn to_string(&self) -> String {
        format!("{}+{}", self.chunk_len, self.checksum_len)
    }

}



pub struct ChecksumPatternMatch {
    pub checksum_pattern: ChecksumPatternSpec,

    #[allow(dead_code)]
    pub offset: i64,
}
