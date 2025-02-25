use crate::error::Result;
use crate::operator::push::base::PipelinePush;
use crate::MetaData;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;

pub struct PrepSquare<R: Runtime> {
	_res: PhantomData<R>,
}

impl<R: Runtime> PipelinePush<R> for PrepSquare<R> {
	type Input<'i> = (&'i MetaData, &'i Handle);
	type Output = (MetaData, Handle);

	fn push<'i>(
		input: Self::Input<'i>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<Self::Output> {
		let (meta, tensor_handle) = input;
		let input_tensor = unsafe {
			TensorHandleRef::<R>::from_raw_parts(
				tensor_handle,
				&meta.stride,
				&meta.shape,
				4,
			)
		};

		let stride = meta.stride.iter().as_slice();

		if stride == &[1, 1] {
			let n = meta.shape[0];
			let output_shape = [n, 2];
			let output_strides = [1, 2];
			let total_bytes = 2 * n * 4;
			let output_handle = client.empty(total_bytes);
			let mut output_tensor = unsafe {
				TensorHandleRef::<R>::from_raw_parts(
					&output_handle,
					&output_strides,
					&output_shape,
					4,
				)
			};

			unsafe {
				prep_square::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new((n * 2) as u32, 1, 1),
					input_tensor.as_tensor_arg(1),
					output_tensor.as_tensor_arg(1),
				);
			}

			let meta = MetaData::build(
				Box::new(output_strides),
				Box::new(output_shape),
			);
			Ok((meta, output_handle))
		} else {
			let m = meta.shape[0];
			let n = meta.shape[1];
			let output_shape = [m, n, 2];
			let output_strides = [n * 2, 2, 1];
			let size = m * n * 2;
			let total_bytes = size * 4;
			let output_handle = client.empty(total_bytes);
			let mut output_tensor = unsafe {
				TensorHandleRef::<R>::from_raw_parts(
					&output_handle,
					&output_strides,
					&output_shape,
					4,
				)
			};

			unsafe {
				prep_square::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new(n as u32, 1, 1),
					input_tensor.as_tensor_arg(1),
					output_tensor.as_tensor_arg(1),
				);
			}
			let meta = MetaData::build(
				Box::new(output_strides),
				Box::new(output_shape),
			);
			Ok((meta, output_handle.clone()))
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::error::Result;
	use crate::MetaData;
	use cubecl::wgpu::WgpuRuntime;

	fn create_meta_vector() -> MetaData {
		MetaData::build(
			Box::new([1, 1]),
			Box::new([6, 1]),
		)
	}

	fn create_meta_matrix() -> MetaData {
		MetaData::build(
			Box::new([6, 1]),
			Box::new([6, 2]),
		)
	}

	#[test]
	fn test_prep_square_vector() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let meta = create_meta_vector();
		let (meta, handle) = meta.handle_from_vec::<WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
		]);

		let (output_meta, output_handle) =
			PrepSquare::<WgpuRuntime>::push((meta, &handle), &client)?;

		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!("Output = {:?}", output_values);
		Ok(())
	}

	#[test]
	fn test_prep_square_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let meta = create_meta_matrix();
		let (meta, handle) = meta.handle_from_vec::<WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);

		let scalar_handle = client.create(f32::as_bytes(&[
			12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
		]));

		let (output_meta, output_handle) =
			PrepSquare::<WgpuRuntime>::push((meta, &handle), &client)?;

		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!("Output = {:?}", output_values);
		Ok(())
	}
}