use crate::error::Result;
use crate::operator::Context;
use crate::MetaData;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// This trait moves the operator after a reduction in the state it needs to be in for the next reduction.
pub trait PipelinePush<R: Runtime> {
	type Input<'i>;
	type Output;

	fn push<'i>(
		input: Self::Input<'i>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<Self::Output>;
}
