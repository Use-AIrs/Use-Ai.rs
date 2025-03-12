use crate::error::Result;
use crate::operator::exec::base::PipelineExec;
use crate::MetaData;

use cubecl::prelude::*;
use cubecl::reduce::instructions::ArgMax;
use cubecl::reduce::reduce;
use cubecl::server::Handle;
use std::marker::PhantomData;

pub struct ExecArgMax<R: Runtime> {
	_phantom: PhantomData<R>,
}

impl<R: Runtime> PipelineExec<R> for ExecArgMax<R> {
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
			reduce::<R, f32, f32, ArgMax>(&client, input, output, axis, None)?;
			let md = MetaData::single();
			Ok((md, output_handle))
		} else {
			let sh = input.shape[1];
			let shape = [1, sh];
			let strides = [1, sh];
			let output_handle = client.empty(sh * 4);
			let output = unsafe {
				TensorHandleRef::<R>::from_raw_parts(&output_handle, &strides, &shape, 4)
			};
			reduce::<R, f32, f32, ArgMax>(&client, input, output, axis, None)?;
			let md = MetaData::build(Box::new(shape), Box::new(strides));
			Ok((md, output_handle))
		}
	}
}
