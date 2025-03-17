#[cfg(test)]
mod tests {
	use crate::error::Result;
	use crate::operator::push::*;
	use cubecl::prelude::*;
	use cubecl::server::Handle;
	use cubecl::wgpu::WgpuRuntime;
	use rand::Rng;
	use std::time::Instant;

	fn create_random_vector(
		client: &ComputeClient<<WgpuRuntime as Runtime>::Server, <WgpuRuntime as Runtime>::Channel>,
		len: usize,
		shape: &'static [usize; 2],
		stride: &'static [usize; 2],
	) -> TensorHandleRef<'static, WgpuRuntime> {
		let mut rng = rand::thread_rng();
		let data: Vec<f32> = (0..len).map(|_| rng.gen()).collect();
		let handle = client.create(f32::as_bytes(&data));
		let static_handle: &'static Handle = Box::leak(Box::new(handle));
		unsafe { TensorHandleRef::from_raw_parts(static_handle, stride, shape, 4) }
	}
	fn create_random_matrix(
		client: &ComputeClient<<WgpuRuntime as Runtime>::Server, <WgpuRuntime as Runtime>::Channel>,
		shape: &'static [usize; 2],
		stride: &'static [usize; 2],
	) -> TensorHandleRef<'static, WgpuRuntime> {
		let total = shape[0] * shape[1];
		let mut rng = rand::thread_rng();
		let data: Vec<f32> = (0..total).map(|_| rng.gen()).collect();
		let handle = client.create(f32::as_bytes(&data));
		let static_handle: &'static Handle = Box::leak(Box::new(handle));
		unsafe { TensorHandleRef::from_raw_parts(static_handle, stride, shape, 4) }
	}

	#[test]
	fn test_prep_square_vector() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let len = 500;
		let shape = Box::leak(Box::new([1, len]));
		let stride = Box::leak(Box::new([1, 1]));
		let dummy_tensor = create_random_vector(&client, len, shape, stride);

		let start = Instant::now();
		let t: TensorHandleRef<WgpuRuntime> = PrepSquare::push((dummy_tensor), &client)?;
		let output_handle = t.handle.clone();

		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		let elapsed = start.elapsed();
		println!(
			"Output = {:?}, Time = {:?}",
			output_values, elapsed
		);
		Ok(())
	}

	#[test]
	fn test_prep_square_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let len = 500;
		let shape = Box::leak(Box::new([1, len]));
		let stride = Box::leak(Box::new([1, 1]));
		let dummy_tensor = create_random_vector(&client, len, shape, stride);
		let start = Instant::now();

		let t: TensorHandleRef<WgpuRuntime> = PrepSquare::push(dummy_tensor, &client)?;
		let output_handle = t.handle.clone();

		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		let elapsed = start.elapsed();
		println!(
			"Output = {:?}, Time = {:?}",
			output_values, elapsed
		);
		Ok(())
	}

	#[test]
	fn test_prep_residuals_vector() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let shape = Box::leak(Box::new([25, 40]));
		let stride = Box::leak(Box::new([40, 1]));
		let dummy_tensor = create_random_matrix(&client, shape, stride);
		let start = Instant::now();

		let t: TensorHandleRef<WgpuRuntime> = PrepSquare::push(dummy_tensor, &client)?;
		let output_handle = t.handle.clone();

		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		let elapsed = start.elapsed();
		println!(
			"Output = {:?}, Time = {:?}",
			output_values, elapsed
		);
		Ok(())
	}

	#[test]
	fn test_prep_residuals_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let shape = Box::leak(Box::new([25, 40]));
		let stride = Box::leak(Box::new([40, 1]));
		let dummy_tensor = create_random_matrix(&client, shape, stride);
		let start = Instant::now();

		let t: TensorHandleRef<WgpuRuntime> = PrepSquare::push(dummy_tensor, &client)?;
		let output_handle = t.handle.clone();
		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		let elapsed = start.elapsed();
		println!(
			"Output = {:?}, Time = {:?}",
			output_values, elapsed
		);
		Ok(())
	}
}
