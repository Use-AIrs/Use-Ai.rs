use mongodb::bson::document::ValueAccessError;
use serde_json::Error as SerdeError;
use thiserror::Error;

pub type Result<T> = core::result::Result<T, StoreError>;

#[derive(Debug, Error)]
pub enum StoreError {
	#[error(transparent)]
	SerdeJsonError(#[from] SerdeError),
	#[error("IO Config Error")]
	IOError(#[from] std::io::Error),
	#[error("Config Version is wrong")]
	InvalidConfig,
	#[error("MangoDB Error")]
	MangoDB(#[from] mongodb::error::Error),
	#[error(transparent)]
	MangoValueAcessError(#[from] ValueAccessError),
	#[error("No Config active")]
	NoConfigActive,
	#[error("Data type not found")]
	NoDataSource,
}
