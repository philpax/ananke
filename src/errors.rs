//! Placeholder; full implementation in Task 1.

#[derive(Debug)]
pub struct ExpectedError;

impl ExpectedError {
    pub fn exit_code(&self) -> u8 {
        1
    }
}

impl std::fmt::Display for ExpectedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "placeholder error")
    }
}

impl std::error::Error for ExpectedError {}
