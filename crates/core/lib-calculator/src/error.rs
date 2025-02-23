use thiserror::Error;

pub type Result<T> = core::result::Result<T, CalcError>;

#[derive(Debug, Error)]
pub enum CalcError {
	#[error("Gpu Error")]
	GpuError,
	#[error("transparent")]
	OperationError,
	#[error("Reduce Error")]
	ReduceError(#[from] cubecl::reduce::ReduceError),
}
