use crate::operator::{Context, Operator};
use cubecl::prelude::*;
use cubecl::Runtime;

mod error;
pub mod gbdt;

pub trait Operation<R: Runtime>: Sized {
	type Ctx: Context;
	type Op: Operator<R>;

	fn exec(
		ctx: Self::Ctx,
		operator: Self::Op,
		client: ComputeClient<R::Server, R::Channel>,
	);
}
