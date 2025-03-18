use crate::error::Result;
use cubecl::prelude::*;

/// This trait is used to execute a reduction instruction.
pub trait PipelineExec<R: Runtime> {
	fn exec<'i, 'o>(
		input: TensorHandleRef<'i, R>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<TensorHandleRef<'o, R>>;
}
