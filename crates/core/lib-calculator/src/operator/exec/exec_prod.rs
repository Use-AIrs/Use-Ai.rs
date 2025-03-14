use crate::error::Result;
use crate::operator::exec::base::PipelineExec;
use crate::MetaData;

use cubecl::prelude::*;
use cubecl::reduce::instructions::Prod;
use cubecl::reduce::reduce;
use cubecl::server::Handle;
use std::marker::PhantomData;

pub struct ExecProd<R: Runtime> {
	_phantom: PhantomData<R>,
}

impl<R: Runtime> PipelineExec<R> for ExecProd<R> {
	fn exec(
		input: TensorHandleRef<R>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<(MetaData, Handle)> {
		if input.shape.len() == 3 {
			let m = input.shape[0];
			let n = input.shape[1];
			let axis = 2;

			let output_shape = [m, n];
			let output_strides = [1, m];
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
				"Prod3d( in: {:?}, out: {:?}",
				&input.shape, &output.shape
			);
			println!();
			reduce::<R, f32, f32, Prod>(&client, input, output, axis, None)?;

			let md = MetaData::build(
				Box::new(output_shape),
				Box::new(output_strides),
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
					"Prod( in: {:?}, out: {:?}",
					&input.shape, &output.shape
				);
				println!();
				reduce::<R, f32, f32, Prod>(&client, input, output, axis, None)?;

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

				println!(
					"Prod( in: {:?}, out: {:?}",
					&input.shape, &output.shape
				);
				println!();
				reduce::<R, f32, f32, Prod>(&client, input, output, axis, None)?;

				let md = MetaData::build(Box::new(shape), Box::new(strides));
				Ok((md, output_handle))
			}
		}
	}
}