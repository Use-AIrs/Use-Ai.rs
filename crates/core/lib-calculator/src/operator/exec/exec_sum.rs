use crate::error::{CalcError, Result};
use crate::operator::exec::base::PipelineExec;
use crate::operator::push::base::PipelinePush;
use cubecl::prelude::*;
use cubecl::reduce::instructions::Sum;
use cubecl::reduce::reduce;
use cubecl::server::Handle;

pub struct ExecSum;

impl<R: Runtime> PipelineExec<R> for ExecSum {
	fn exec(
		input: TensorHandleRef<R>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<Handle> {
		let axis = if input.strides == [1, 1] { 1 } else { 0 };
		if axis == 1 {
			let output_handle = client.empty(4);
			let output = unsafe {
				TensorHandleRef::<R>::from_raw_parts(
					&output_handle,
					&[1, 1],
					&[1, 1],
					4,
				)
			};
			let res = reduce::<R, f32, f32, Sum>(
				&client, input, output, axis, None,
			);
			if res.is_ok() {
				Ok(output_handle)
			} else {
				Err(CalcError::GpuError)
			}
		} else {
			let n = input.shape[1];
			let shape = [1, n];
			let strides = [1, n];
			let output_handle = client.empty(n * 4);
			let output = unsafe {
				TensorHandleRef::<R>::from_raw_parts(
					&output_handle,
					&strides,
					&shape,
					4,
				)
			};
			reduce::<R, f32, f32, Sum>(
				&client, input, output, axis, None,
			)?;
			Ok(output_handle)
		}
	}
}