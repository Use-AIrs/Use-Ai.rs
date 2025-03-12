use crate::error::Result;
use crate::operator::exec::base::PipelineExec;
use crate::MetaData;

use cubecl::prelude::*;
use cubecl::reduce::instructions::Sum;
use cubecl::reduce::reduce;
use cubecl::server::Handle;
use std::marker::PhantomData;

pub struct ExecSum<R: Runtime> {
	_phantom: PhantomData<R>,
}

impl<R: Runtime> PipelineExec<R> for ExecSum<R> {
	fn exec(
		input: TensorHandleRef<R>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<(MetaData, Handle)> {
		let axis = if input.strides == [1, 1] { 1 } else { 0 };
		if axis == 1 {
			let output_handle = client.empty(4);
			let output = unsafe {
				TensorHandleRef::<R>::from_raw_parts(&output_handle, &[1, 1], &[1, 1], 4)
			};
			reduce::<R, f32, f32, Sum>(&client, input, output, axis, None)?;
			let md = MetaData::single();
			Ok((md, output_handle))
		} else {
			let n = input.shape[1];
			let shape = [1, n];
			let strides = [1, n];
			let output_handle = client.empty(n * 4);
			let output = unsafe {
				TensorHandleRef::<R>::from_raw_parts(&output_handle, &strides, &shape, 4)
			};
			reduce::<R, f32, f32, Sum>(&client, input, output, axis, None)?;
			let md = MetaData::build(Box::new(shape), Box::new(strides));
			Ok((md, output_handle))
		}
	}
}
