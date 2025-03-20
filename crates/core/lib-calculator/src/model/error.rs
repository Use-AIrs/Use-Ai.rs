use thiserror::Error;

pub type Result<T> = core::result::Result<T, ModelError>;

#[derive(Debug, Error)]
pub enum ModelError {}
