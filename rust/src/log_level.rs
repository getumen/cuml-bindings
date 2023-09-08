#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Off = 0,
    Critical = 1,
    Error = 2,
    Warn = 3,
    Info = 4,
    Debug = 5,
    Trace = 6,
}
