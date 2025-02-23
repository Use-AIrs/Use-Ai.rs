use thiserror::Error;

pub type Result<T> = core::result::Result<T, OpError>;

#[derive(Debug, Error)]
pub enum OpError {
	#[error("GBDT")]
	OpError,
}
