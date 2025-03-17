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
		if input.shape.len() == 3 {
			let m = input.shape[0];
			let n = input.shape[1];
			let axis = 2;

			let output_shape = [m, n];
			let output_strides = [n, 1];
			let output_handle = client.empty(m * n * 4);

			let output = unsafe {
				TensorHandleRef::<R>::from_raw_parts(
					&output_handle,
					&output_strides,
					&output_shape,
					4,
				)
			};
			println!();
			println!(
				"ArgMax3d( in: {:?}, out: {:?}",
				&input.shape, &output.shape
			);
			reduce::<R, f32, f32, ArgMax>(&client, input, output, axis, None)?;

			let md = MetaData::build(
				Box::new(output_strides),
				Box::new(output_shape),
			);
			Ok((md, output_handle))
		} else {
			let axis = if input.strides == [1, 1] { 1 } else { 0 };
			if axis == 1 {
				let output_handle = client.empty(4);
				let output = unsafe {
					TensorHandleRef::<R>::from_raw_parts(&output_handle, &[1, 1], &[1, 1], 4)
				};
				println!();
				println!(
					"ArgMax( in: {:?}, out: {:?}",
					&input.shape, &output.shape
				);
				reduce::<R, f32, f32, ArgMax>(&client, input, output, axis, None)?;
				let md = MetaData::single();
				Ok((md, output_handle))
			} else {
				let n = input.shape[0];
				let shape = [n, 1];
				let strides = [1, 1];
				let output_handle = client.empty(n * 4);
				let output = unsafe {
					TensorHandleRef::<R>::from_raw_parts(&output_handle, &strides, &shape, 4)
				};
				println!();
				println!(
					"ArgMax( in: {:?}, out: {:?}",
					&input.shape, &output.shape
				);
				reduce::<R, f32, f32, ArgMax>(&client, input, output, 1, None)?;
				let md = MetaData::build(Box::new(strides), Box::new(shape));
				Ok((md, output_handle))
			}
		}
	}
}
