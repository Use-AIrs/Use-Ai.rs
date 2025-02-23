use crate::error::Result;
use crate::operator::push::base::PipelinePush;
use crate::MetaData;
use cubecl::prelude::*;
use cubecl::server::Handle;
/*
/// Takes a Scalar what is in the gpu and builds a Vector with the allocated Memory
pub struct PrepResiduals;

impl<R: Runtime> PipelinePush<R> for PrepResiduals {
    fn push(
        input:(&MetaData, &Handle),
        client: ComputeClient<R::Server, R::Channel>,
    ) -> Result<()> {
        let (meta, handle) = input;
        let mut tensor = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                handle,
                &meta.stride,
                &meta.shape,
                4,
            )
        };
        let mut buffer = ScalarArg::new(0.0 as f32);
        unsafe {
            prep_residuals::launch_unchecked::<f32, R>(
                &mut tensor.as_tensor_arg(1),
                &mut buffer,
                &client,
            );
        }

        Ok(())
    }


}

#[cube(launch_unchecked)]
pub fn prep_residuals<T: Numeric>(
    t: &mut Tensor<T>,
    buffer: &mut ScalarArg<T>
) {
    buffer = t[0];
    let end = t.len();
    #[unroll]
    for i in 0..end {
        t[i] = - buffer;
    }
}
*/
