use crate::error::Result;
use cubecl::prelude::*;

/// This trait moves the operator after a reduction in the state it needs to be in for the next reduction.
pub trait PipelinePush<R: Runtime> {
	type Input<'i>;

	fn push<'i, 'o>(
		input: Self::Input<'i>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<TensorHandleRef<'o, R>>;
}
