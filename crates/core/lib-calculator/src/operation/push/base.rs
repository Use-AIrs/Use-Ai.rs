use crate::ctx::CpuCtx;
use crate::operation::gpu_ctx::GpuCtx;
use cubecl::Runtime;

pub trait PipelinePush<R: Runtime> {
    type Output;

    fn init(ctx: GpuCtx<R>) -> Self::Output;
    fn push(self) -> Self::Output;
}
