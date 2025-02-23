use crate::error::Result;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// This trait is used to execute a reduction instruction.
pub trait PipelineExec<R: Runtime> {
	fn exec(
		input: TensorHandleRef<R>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<Handle>;
}
