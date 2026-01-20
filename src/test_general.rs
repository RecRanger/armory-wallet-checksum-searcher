use std::sync::OnceLock;

#[derive(Debug)]
pub struct TestConfig {
    /// Set with environment variable `TEST_MAX_DATA_SIZE_MIB=128`, for example.
    pub max_data_size_mebibytes: usize,

    /// Set with environment variable `TEST_ENABLE_SLOW_TESTS=true`.
    pub enable_slow_tests: bool,
}

static TEST_CONFIG: OnceLock<TestConfig> = OnceLock::new();

pub fn get_test_config() -> &'static TestConfig {
    TEST_CONFIG.get_or_init(|| {
        let max_data_size_mebibytes = std::env::var("TEST_MAX_DATA_SIZE_MIB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);

        let enable_slow_tests = std::env::var("TEST_ENABLE_SLOW_TESTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        TestConfig {
            max_data_size_mebibytes,
            enable_slow_tests,
        }
    })
}
