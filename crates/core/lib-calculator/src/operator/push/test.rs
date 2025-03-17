#[cfg(test)]
mod tests {
	use crate::error::Result;
	use crate::operator::push::*;
	use crate::MetaData;
	use cubecl::prelude::*;
	use cubecl::wgpu::WgpuRuntime;
	use std::time::Instant;

	fn create_meta_vector() -> MetaData {
		MetaData::build(Box::new([1, 1]), Box::new([1, 6]))
	}

	fn create_meta_matrix() -> MetaData {
		MetaData::build(Box::new([6, 1]), Box::new([2, 6]))
	}

	#[test]
	fn test_prep_square_vector() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let meta = create_meta_vector();
		let cpu_tensor = meta.cputensor_from_vec::<WgpuRuntime>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

		let start = Instant::now();
		let t: TensorHandleRef<WgpuRuntime> = PrepSquare::push(
			(&cpu_tensor.meta, &cpu_tensor.handle),
			&client,
		)?;
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
		let meta = create_meta_matrix();
		let cpu_tensor = meta.cputensor_from_vec::<WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);
		let start = Instant::now();

		let t: TensorHandleRef<WgpuRuntime> = PrepSquare::push(
			(&cpu_tensor.meta, &cpu_tensor.handle),
			&client,
		)?;
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
		let meta = create_meta_vector();
		let cpu_tensor = meta.cputensor_from_vec::<WgpuRuntime>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

		let scalar_handle = client.create(f32::as_bytes(&[3.0]));
		let start = Instant::now();

		let t: TensorHandleRef<WgpuRuntime> = PrepResiduals::push(
			(
				&cpu_tensor.meta,
				&cpu_tensor.handle,
				&scalar_handle,
			),
			&client,
		)?;

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
		let meta = create_meta_matrix();
		let cpu_tensor = meta.cputensor_from_vec::<WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);

		let scalar_handle = client.create(f32::as_bytes(&[
			12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
		]));
		let start = Instant::now();

		let t: TensorHandleRef<WgpuRuntime> = PrepResiduals::push(
			(
				&cpu_tensor.meta,
				&cpu_tensor.handle,
				&scalar_handle,
			),
			&client,
		)?;
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
