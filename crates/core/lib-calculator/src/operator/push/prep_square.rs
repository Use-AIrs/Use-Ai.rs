use crate::error::Result;
use crate::operator::push::base::PipelinePush;
use cubecl::prelude::*;
use std::marker::PhantomData;

pub struct PrepSquare<R: Runtime> {
	_phantom: PhantomData<R>,
}

impl<R: Runtime> PipelinePush<R> for PrepSquare<R> {
	type Input<'i> = TensorHandleRef<'i, R>;

	fn push<'i, 'o>(
		input: Self::Input<'i>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<TensorHandleRef<'o, R>> {
		let stride = input.strides.iter().as_slice();
		if stride == &[1, 1] {
			let n = input.shape[1];
			let output_shape = Box::leak(Box::new([2, n]));
			let output_strides = Box::leak(Box::new([n, 1]));
			let total_bytes = 2 * n * 4;
			let output_handle = Box::leak(Box::new(client.empty(total_bytes)));
			let output_tensor = unsafe {
				TensorHandleRef::<'_, R>::from_raw_parts(
					output_handle,
					output_strides,
					output_shape,
					4,
				)
			};
			println!(
				"PrepSquare( in: {:?}, out: {:?}",
				&input.shape, &output_tensor.shape
			);
			unsafe {
				prep_square::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new((n * 2) as u32, 1, 1),
					input.as_tensor_arg(1),
					output_tensor.as_tensor_arg(1),
				);
			}
			let output_tensor = unsafe {
				TensorHandleRef::<'o, R>::from_raw_parts(
					output_handle,
					output_strides,
					output_shape,
					4,
				)
			};
			Ok(output_tensor)
		} else {
			let m = input.shape[0];
			let n = input.shape[1];
			let output_shape = Box::leak(Box::new([m, n, 2]));
			let output_strides = Box::leak(Box::new([n, m, 1]));
			let size = m * n * 2;
			let total_bytes = size * 4;
			let output_handle = Box::leak(Box::new(client.empty(total_bytes)));
			let output_tensor = unsafe {
				TensorHandleRef::<'_, R>::from_raw_parts(
					output_handle,
					output_strides,
					output_shape,
					4,
				)
			};
			println!(
				"PrepSquare( in: {:?}, out: {:?}",
				&input.shape, &output_tensor.shape
			);
			unsafe {
				prep_square::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new(n as u32, 1, 1),
					input.as_tensor_arg(1),
					output_tensor.as_tensor_arg(1),
				);
			}
			let output_tensor = unsafe {
				TensorHandleRef::<'o, R>::from_raw_parts(
					output_handle,
					output_strides,
					output_shape,
					4,
				)
			};
			Ok(output_tensor)
		}
	}
}

#[cube(launch_unchecked)]
pub fn prep_square<T: Numeric>(
	input: &Tensor<T>,
	output: &mut Tensor<T>,
) {
	let len = output.len();
	let cut = len / 2u32;
	for i in 0..cut {
		output[i] = input[i];
	}
	for i in cut..len {
		output[i] = input[i - cut];
	}
}
