use thiserror::Error;

#[derive(Debug, Error)]
pub enum CumlError {
    #[error(transparent)]
    CError(anyhow::Error),
}

impl From<anyhow::Error> for CumlError {
    fn from(error: anyhow::Error) -> Self {
        CumlError::CError(error)
    }
}
