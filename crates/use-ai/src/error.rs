use inquire::InquireError;
use lib_store::error::StagingError;
use thiserror::Error;

pub type Result<T> = core::result::Result<T, ToolError>;

#[derive(Debug, Error)]
pub enum ToolError {
	#[error(transparent)]
	UseAiMenuError(#[from] InquireError),
	#[error(transparent)]
	StagingError(#[from] StagingError),
	#[error("Unknown Error")]
	UseAiError,
}
