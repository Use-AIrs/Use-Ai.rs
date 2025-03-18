use crate::error::Result;
use crate::operator::push::base::PipelinePush;
use cubecl::prelude::*;
use std::marker::PhantomData;

pub struct PrepResiduals<R: Runtime> {
	_phantom: PhantomData<R>,
}

impl<R: Runtime> PipelinePush<R> for PrepResiduals<R> {
	type Input<'i> = (
		TensorHandleRef<'i, R>,
		TensorHandleRef<'i, R>,
	);

	fn push<'i, 'o>(
		input: Self::Input<'i>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<TensorHandleRef<'o, R>> {
		let (t0, t1) = input;
		let t1_len = t1.size();
		if t1_len == 4 {
			let n = t0.shape[1];
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
				"PrepRes( in: {:?}, out: {:?}",
				&t0.shape, &output_tensor.shape
			);
			unsafe {
				prep_residuals::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new((n * 2) as u32, 1, 1),
					t0.as_tensor_arg(1),
					t1.as_tensor_arg(1),
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
			let m = t0.shape[0];
			let n = t0.shape[1];
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
				"PrepRes( in: {:?}, out: {:?}",
				&t0.shape, &output_tensor.shape
			);
			unsafe {
				prep_residuals::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new(n as u32, 1, 1),
					t0.as_tensor_arg(1),
					t1.as_tensor_arg(1),
					output_tensor.as_tensor_arg(1),
				);
			}
			let output_tensor = unsafe {
				TensorHandleRef::<'_, R>::from_raw_parts(
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
pub fn prep_residuals<T: Numeric>(
	input: &Tensor<T>,
	neg: &Tensor<T>,
	output: &mut Tensor<T>,
) {
	let zero = T::from_int(0);
	let a = input.stride(0);
	let cut = input.len();
	let len = output.len();
	for i in 0..a {
		output[i] = input[i];
	}
	let mut g = 0;
	for i in cut..len {
		output[i] = zero - neg[g];
		if g % a == 0 {
			g += 1;
		}
	}
}
