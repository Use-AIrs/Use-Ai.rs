use crate::operation::{Context, Operator};
use cubecl::prelude::*;
use cubecl::Runtime;

mod error;
pub mod gbdt;

pub trait Operation<R: Runtime>: Sized {
    type Ctx: Context;
    type Op: Operator<R>;
    type Output;

    fn exec(
        ctx: Self::Ctx,
        operator: Self::Op,
        client: ComputeClient<R::Server, R::Channel>,
    ) -> Self::Output;
}
