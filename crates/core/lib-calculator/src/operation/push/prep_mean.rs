/*use cubecl::Runtime;
use crate::ctx::CpuCtx;
use crate::operation::{Operator, OperatorResult};
use crate::operation::push::base::PipelinePush;
use crate::operation::gpu_ctx::GpuCtx;

pub struct PrepMean;

impl<R: Runtime>  PipelinePush for PrepMean {
    type Output = OperatorResult<R>;

    fn init(ctx: &GpuCtx<R>) -> Self::Output {
        todo!(ctx)
    }

    fn push(self) -> Self::Output {
        todo!()
    }
}
*/