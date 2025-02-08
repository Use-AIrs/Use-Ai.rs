use crate::error::Result;
use cubecl::prelude::*;

/// This trait is used to execute a reduction instruction.
pub trait PipelineExec<R: Runtime> {
    fn exec(
        op: TensorHandleRef<'_, R>,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<()>;
}
