/*
#[cfg(test)]
mod tests {
	use crate::error::Result;
	use crate::operator::*;
	use crate::MetaData;
	use action_space::action_space;
	use cubecl::wgpu::WgpuRuntime;

	fn create_meta_vector() -> MetaData {
		MetaData::build(Box::new([1, 1]), Box::new([1, 12]))
	}

	fn create_meta_matrix() -> MetaData {
		MetaData::build(Box::new([6, 1]), Box::new([2, 6]))
	}

	#[test]
	fn test_action_space_vector<R: Runtime>() -> Result<()> {
		let meta = create_meta_vector();
		let tensor = meta.tensorref_from_vec::<R>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);

		let score = action_space!(
			(tensor, ExecMean, output_handle),
			((tensor, output_handle), PrepResiduals),
			(ExecSum),
			(PrepSquare),
			(ExecProd),
		);

		let client = WgpuRuntime::client(&Default::default());
		let binding = score.handle.clone().binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!();
		println!("Output = {:?}", output_values);
		Ok(())
	}

	#[test]
	fn test_action_space_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let meta = create_meta_matrix();
		let tensor = meta.tensorref_from_vec::<'static, WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);

		let score = action_space!(
			(tensor, ExecMean, output_handle),
			((tensor, output_handle), PrepResiduals),
			(ExecSum),
			(PrepSquare),
			(ExecProd),
		);

		let binding = score.handle.clone().binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!();
		println!("Output = {:?}", output_values);
		Ok(())
	}
}
*/
