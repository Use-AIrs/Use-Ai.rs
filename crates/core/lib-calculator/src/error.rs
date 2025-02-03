use thiserror::Error;

pub type Result<T> = core::result::Result<T, CalcError>;

#[derive(Debug, Error)]
pub enum CalcError {
    #[error("Gpu Error")]
    GpuError,
}
