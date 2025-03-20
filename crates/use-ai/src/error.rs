use inquire::InquireError;
use lib_stage::error::StageError;
use lib_store::error::StoreError;
use thiserror::Error;

pub type Result<T> = core::result::Result<T, ToolError>;

#[derive(Debug, Error)]
pub enum ToolError {
	#[error(transparent)]
	UseAiMenuError(#[from] InquireError),
	#[error(transparent)]
	StageError(#[from] StageError),
	#[error(transparent)]
	StoreError(#[from] StoreError),
}
