use crate::error::Result;
use crate::operator::exec::base::PipelineExec;

use cubecl::prelude::*;
use cubecl::reduce::instructions::ArgMax;
use cubecl::reduce::reduce;
use std::marker::PhantomData;

pub struct ExecArgMax<R: Runtime> {
	_phantom: PhantomData<R>,
}

impl<R: Runtime> PipelineExec<R> for ExecArgMax<R> {
	fn exec<'i, 'o>(
		input: TensorHandleRef<'i, R>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<TensorHandleRef<'o, R>> {
		if input.shape.len() == 3 {
			let m = input.shape[0];
			let n = input.shape[1];
			let axis = 2;

			let output_shape = Box::leak(Box::new([m, n]));
			let output_strides = Box::leak(Box::new([n, 1]));
			let output_handle = Box::leak(Box::new(client.empty(m * n * 4)));

			let output = unsafe {
				TensorHandleRef::<R>::from_raw_parts(
					output_handle,
					output_strides,
					output_shape,
					4,
				)
			};
			println!();
			println!(
				"ArgMax3d( in: {:?}, out: {:?})",
				&input.shape, &output.shape
			);
			reduce::<R, f32, f32, ArgMax>(client, input, output, axis, None)?;

			let output =
				unsafe { TensorHandleRef::<R>::from_raw_parts(output_handle, &[1, 1], &[1, 1], 4) };

			Ok(output)
		} else {
			if input.strides == [1, 1] {
				let output_handle = Box::leak(Box::new(client.empty(4)));

				let output = unsafe {
					TensorHandleRef::<R>::from_raw_parts(output_handle, &[1, 1], &[1, 1], 4)
				};
				println!();
				println!(
					"ArgMax( in: {:?}, out: {:?})",
					&input.shape, &output.shape
				);
				reduce::<R, f32, f32, ArgMax>(client, input, output, 1, None)?;

				let output = unsafe {
					TensorHandleRef::<R>::from_raw_parts(output_handle, &[1, 1], &[1, 1], 4)
				};

				Ok(output)
			} else {
				let n = input.shape[0];
				let shape = Box::leak(Box::new([n, 1]));
				let strides = Box::leak(Box::new([1, 1]));
				let output_handle = Box::leak(Box::new(client.empty(n * 4)));

				let output = unsafe {
					TensorHandleRef::<R>::from_raw_parts(output_handle, strides, shape, 4)
				};
				println!();
				println!(
					"ArgMax( in: {:?}, out: {:?})",
					&input.shape, &output.shape
				);
				reduce::<R, f32, f32, ArgMax>(client, input, output, 1, None)?;

				let output = unsafe {
					TensorHandleRef::<R>::from_raw_parts(output_handle, strides, shape, 4)
				};

				Ok(output)
			}
		}
	}
}
