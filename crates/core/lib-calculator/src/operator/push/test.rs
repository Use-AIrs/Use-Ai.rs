#[cfg(test)]
mod tests {
	use crate::error::Result;
	use crate::operator::push::*;
	use cubecl::prelude::*;
	use cubecl::server::Handle;
	use cubecl::wgpu::WgpuRuntime;
	use rand::Rng;
	use std::time::Instant;

	fn create_random(
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
		let len = 6;
		let shape = Box::leak(Box::new([1, len]));
		let stride = Box::leak(Box::new([len, 1]));
		let dummy_tensor = create_random(&client, shape, stride);

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
		println!();
		Ok(())
	}

	#[test]
	fn test_prep_square_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let shape = Box::leak(Box::new([4, 10]));
		let stride = Box::leak(Box::new([10, 1]));
		let dummy_tensor = create_random(&client, shape, stride);
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
		println!();
		Ok(())
	}

	#[test]
	fn test_prep_residuals_vector() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let len = 6;
		let shape = Box::leak(Box::new([1, len]));
		let stride = Box::leak(Box::new([len, 1]));
		let dummy_tensor = create_random(&client, shape, stride);
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
		println!();
		Ok(())
	}

	#[test]
	fn test_prep_residuals_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let shape = Box::leak(Box::new([4, 10]));
		let stride = Box::leak(Box::new([10, 1]));
		let dummy_tensor = create_random(&client, shape, stride);
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
		println!();
		Ok(())
	}

	#[test]
	fn test_prep_square_vector_debug() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let len = 4;
		let shape = Box::leak(Box::new([1, len]));
		let stride = Box::leak(Box::new([len, 1]));

		let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
		let handle = client.create(f32::as_bytes(&data));
		let static_handle: &'static Handle = Box::leak(Box::new(handle));
		let dummy_tensor =
			unsafe { TensorHandleRef::from_raw_parts(static_handle, stride, shape, 4) };

		println!("Input vector = [1.0, 2.0, 3.0, 4.0]");

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
		println!();
		Ok(())
	}

	#[test]
	fn test_prep_square_matrix_debug() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let shape = Box::leak(Box::new([2, 3]));
		let stride = Box::leak(Box::new([3, 1]));

		let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
		let handle = client.create(f32::as_bytes(&data));
		let dummy_tensor = unsafe { TensorHandleRef::from_raw_parts(&handle, stride, shape, 4) };

		println!("Input matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]");

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
		println!();
		Ok(())
	}

	#[test]
	fn test_prep_residuals_vector_debug() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let len = 4;
		let shape = Box::leak(Box::new([1, len]));
		let stride = Box::leak(Box::new([len, 1]));

		let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
		let handle = client.create(f32::as_bytes(&data));
		let dummy_tensor = unsafe { TensorHandleRef::from_raw_parts(&handle, stride, shape, 4) };

		println!("Input vector = [1.0, 2.0, 3.0, 4.0]");

		let data_vec = vec![2.0];
		let handle_vec = client.create(f32::as_bytes(&data_vec));
		let static_handle_vec = Box::leak(Box::new(handle_vec));
		let dummy_tensor_vec =
			unsafe { TensorHandleRef::from_raw_parts(static_handle_vec, stride, shape, 4) };

		let start = Instant::now();
		let t: TensorHandleRef<WgpuRuntime> =
			PrepResiduals::push((dummy_tensor, dummy_tensor_vec), &client)?;
		let output_handle = t.handle.clone();
		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		let elapsed = start.elapsed();

		println!(
			"Output = {:?}, Time = {:?} UB",
			output_values, elapsed
		);
		println!();
		Ok(())
	}

	#[test]
	fn test_prep_residuals_matrix_debug() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());

		let shape = Box::leak(Box::new([2, 3]));
		let stride = Box::leak(Box::new([3, 1]));

		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
		let handle = client.create(f32::as_bytes(&data));
		let static_handle = Box::leak(Box::new(handle));
		let dummy_tensor =
			unsafe { TensorHandleRef::from_raw_parts(static_handle, stride, shape, 4) };

		let data_vec = vec![1.5, 2.0];
		let handle_vec = client.create(f32::as_bytes(&data_vec));
		let static_handle_vec = Box::leak(Box::new(handle_vec));
		let dummy_tensor_vec =
			unsafe { TensorHandleRef::from_raw_parts(static_handle_vec, stride, shape, 4) };
		println!("Input matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]");

		let start = Instant::now();
		let t: TensorHandleRef<WgpuRuntime> =
			PrepResiduals::push((dummy_tensor, dummy_tensor_vec), &client)?;
		let output_handle = t.handle.clone();
		let binding = output_handle.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		let elapsed = start.elapsed();

		println!(
			"Output = {:?}, Time = {:?} UB",
			output_values, elapsed
		);
		println!();
		Ok(())
	}
}
