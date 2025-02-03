use lib_stage::error::TransformerError;
use mongodb::bson::document::ValueAccessError;
use serde_json::Error as SerdeError;
use thiserror::Error;

pub type Result<T> = core::result::Result<T, StagingError>;

#[derive(Debug, Error)]
pub enum StagingError {
    #[error(transparent)]
    SerdeJsonError(#[from] SerdeError),
    #[error("IO Config Error")]
    IOError(#[from] std::io::Error),
    #[error("Config Version is wrong")]
    InvalidConfig,
    #[error("MangoDB Error")]
    MangoDB(#[from] mongodb::error::Error),
    #[error("MangoDB Value Access Error")]
    MangoValueAcessError(#[from] ValueAccessError),
    #[error("No Config active")]
    NoConfigActive,
    #[error("Data type not found")]
    NoDataSource,
    #[error("Data type not found")]
    NoDataScheme(#[from] TransformerError),
}
