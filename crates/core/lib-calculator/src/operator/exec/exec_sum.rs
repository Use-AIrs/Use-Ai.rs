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

			println!(
				"Sum3d( in: {:?}, out: {:?}",
				&input.shape, &output.shape
			);
			println!();
			reduce::<R, f32, f32, Sum>(&client, input, output, axis, None)?;

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
				println!(
					"Sum( in: {:?}, out: {:?}",
					&input.shape, &output.shape
				);
				println!();
				reduce::<R, f32, f32, Sum>(&client, input, output, axis, None)?;
				let md = MetaData::single();
				Ok((md, output_handle))
			} else {
				let n = input.shape[1];
				let shape = [1, n];
				let strides = [n, 1];
				let output_handle = client.empty(n * 4);
				let output = unsafe {
					TensorHandleRef::<R>::from_raw_parts(&output_handle, &strides, &shape, 4)
				};
				println!(
					"Sum( in: {:?}, out: {:?}",
					&input.shape, &output.shape
				);
				println!();
				reduce::<R, f32, f32, Sum>(&client, input, output, axis, None)?;
				let md = MetaData::build(Box::new(strides), Box::new(shape));
				Ok((md, output_handle))
			}
		}
	}
}