use ndarray::ShapeError;
use serde_json::Error as SerdeError;
use std::convert::Infallible;
use std::num::{ParseFloatError, ParseIntError};
use thiserror::Error;

pub type Result<T> = core::result::Result<T, TransformerError>;

#[derive(Debug, Error)]
pub enum TransformerError {
    #[error("CSV Prase Error")]
    CsvError,
    #[error("CSV Array Error: {0}")]
    CsvArrayError(String),
    #[error("CSV to Data Raw Error")]
    CsvToDataRawError,
    #[error(transparent)]
    ConversionError(#[from] Infallible),
    #[error(transparent)]
    NoHeaderConversionError(#[from] ParseFloatError),
    #[error("Transform Not Featured")]
    TransformNotFeatured,
    #[error("Categorize Error")]
    CategorizeError,
    #[error("No Header")]
    NoHeader,
    #[error("Category Error: {0}")]
    CategoryError(String),
    #[error(transparent)]
    ParseUsizeError(#[from] ParseIntError),
    #[error("HashMap Error")]
    HashMapError,
    #[error("Array shaping Error")]
    ShapingError(#[from] ShapeError),
    #[error("No input columns")]
    NoInputColumns,
    #[error("Data type isnt supported")]
    DataTypeNotSupported,
    #[error("No Transformations")]
    NoTransformations,
    #[error("No Transformation Parameters")]
    NoTransformationParams,
    #[error("Transformation Parameters didnt work")]
    TransformationParamsWrong,
    #[error("No Transformations Defined")]
    NoTransformationsDefined,
}
