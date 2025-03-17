use crate::error::Result;
use crate::operator::push::base::PipelinePush;
use crate::MetaData;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;

pub struct PrepResiduals<R: Runtime> {
	_phantom: PhantomData<R>,
}

impl<R: Runtime> PipelinePush<R> for PrepResiduals<R> {
	type Input<'i> = (&'i MetaData, &'i Handle, &'i Handle);

	fn push<'i, 'o>(
		input: Self::Input<'i>,
		client: &ComputeClient<R::Server, R::Channel>,
	) -> Result<TensorHandleRef<'o, R>> {
		let (meta, tensor_handle, scalar_handle) = input;
		let input_tensor = unsafe {
			TensorHandleRef::<'i, R>::from_raw_parts(
				tensor_handle,
				&meta.stride,
				&meta.shape,
				4,
			)
		};

		let stride = meta.stride.iter().as_slice();

		let (output_handle, output_strides, output_shape) = if stride == &[1, 1] {
			let n = meta.shape[0];
			let output_shape = [n, 2];
			let output_strides = [1, 2];
			let total_bytes = 2 * n * 4;
			let output_handle = client.empty(total_bytes);

			let binding = scalar_handle.clone().binding();
			let bytes = client.read_one(binding);
			let output_value = f32::from_bytes(&bytes);
			let value = output_value[0];

			let vec = vec![value; n];
			let neg_handle = client.create(f32::as_bytes(vec.as_slice()));
			let neg_tensor = unsafe {
				TensorHandleRef::<'_, R>::from_raw_parts(&neg_handle, &meta.stride, &meta.shape, 4)
			};

			let output_tensor_for_kernel = unsafe {
				TensorHandleRef::<'_, R>::from_raw_parts(
					&output_handle,
					&output_strides,
					&output_shape,
					4,
				)
			};

			println!();
			println!(
				"PrepRes( in: {:?}, out: {:?}",
				&input_tensor.shape, &output_tensor_for_kernel.shape
			);

			unsafe {
				prep_residuals::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new((n * 2) as u32, 1, 1),
					input_tensor.as_tensor_arg(1),
					neg_tensor.as_tensor_arg(1),
					output_tensor_for_kernel.as_tensor_arg(1),
				);
			}

			(
				output_handle,
				output_strides.to_vec(),
				output_shape.to_vec(),
			)
		} else {
			let m = meta.shape[0];
			let n = meta.shape[1];
			let output_shape = [m, n, 2];
			let output_strides = [n * 2, 2, 1];
			let size = m * n * 2;
			let total_bytes = size * 4;

			let output_handle = client.empty(total_bytes);

			let output_tensor_for_kernel = unsafe {
				TensorHandleRef::<'_, R>::from_raw_parts(
					&output_handle,
					&output_strides,
					&output_shape,
					4,
				)
			};

			let neg_shape = [m, 1];
			let neg_strides = [1, 1];
			let neg_tensor = unsafe {
				TensorHandleRef::<'_, R>::from_raw_parts(scalar_handle, &neg_strides, &neg_shape, 4)
			};

			println!();
			println!(
				"PrepRes( in: {:?}, out: {:?}",
				&input_tensor.shape, &output_tensor_for_kernel.shape
			);

			unsafe {
				prep_residuals::launch_unchecked::<f32, R>(
					&client,
					CubeCount::Static(1, 1, 1),
					CubeDim::new(n as u32, 1, 1),
					input_tensor.as_tensor_arg(1),
					neg_tensor.as_tensor_arg(1),
					output_tensor_for_kernel.as_tensor_arg(1),
				);
			}
			(
				output_handle,
				output_strides.to_vec(),
				output_shape.to_vec(),
			)
		};

		let output_tensor = unsafe {
			let handle_ref: &'o Handle = std::mem::transmute(&output_handle);
			let strides_ref: &'o [usize] = std::mem::transmute(output_strides.as_slice());
			let shape_ref: &'o [usize] = std::mem::transmute(output_shape.as_slice());

			TensorHandleRef::<'o, R>::from_raw_parts(handle_ref, strides_ref, shape_ref, 4)
		};

		Ok(output_tensor)
	}
}

#[cube(launch_unchecked)]
pub fn prep_residuals<T: Float>(
	input: &Tensor<T>,
	neg: &Tensor<T>,
	output: &mut Tensor<T>,
) {
	let len = output.len();
	let cut = len / 2u32;
	for i in 0..cut {
		output[i] = input[i];
	}

	for i in cut..len {
		output[i] = neg[i - cut];
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::error::Result;
	use crate::MetaData;
	use cubecl::wgpu::WgpuRuntime;

	fn create_meta_vector() -> MetaData {
		MetaData::build(Box::new([1, 1]), Box::new([6, 1]))
	}

	fn create_meta_matrix() -> MetaData {
		MetaData::build(Box::new([6, 1]), Box::new([6, 2]))
	}

	#[test]
	fn test_prep_residuals_vector() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let meta = create_meta_vector();
		let (meta, handle) =
			meta.handle_from_vec::<WgpuRuntime>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

		let scalar_handle = client.create(f32::as_bytes(&[3.0]));

		let t: TensorHandleRef<WgpuRuntime> =
			PrepResiduals::push((meta, &handle, &scalar_handle), &client)?;

		let output_handle = t.handle.clone();

		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!("Output = {:?}", output_values);
		Ok(())
	}

	#[test]
	fn test_prep_residuals_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let meta = create_meta_matrix();
		let (meta, handle) = meta.handle_from_vec::<WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);

		let scalar_handle = client.create(f32::as_bytes(&[
			12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
		]));

		let t: TensorHandleRef<WgpuRuntime> =
			PrepResiduals::push((meta, &handle, &scalar_handle), &client)?;
		let output_handle = t.handle.clone();
		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!("Output = {:?}", output_values);
		Ok(())
	}
}
