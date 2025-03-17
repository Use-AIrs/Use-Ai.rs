#[cfg(test)]
mod tests {
	use crate::error::Result;
	use crate::operator::*;
	use crate::utils::to_tref;
	use crate::MetaData;
	use cubecl::wgpu::WgpuRuntime;
	use lib_proc_macros::action_space;

	fn create_meta_vector() -> MetaData {
		MetaData::build(Box::new([1, 1]), Box::new([1, 12]))
	}

	fn create_meta_matrix() -> MetaData {
		MetaData::build(Box::new([6, 1]), Box::new([2, 6]))
	}

	#[test]
	fn test_action_space_vector() -> Result<()> {
		let meta = create_meta_vector();
		let (meta_input, handle_input) = meta.handle_from_vec::<WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);
		let some_tensor = to_tref::<WgpuRuntime>((&meta_input, &handle_input));

		let score = action_space!(
			WgpuRuntime,
			(some_tensor, ExecMean, some_output),
			(
				(&meta_input, &handle_input, &some_output),
				PrepResiduals
			),
			(ExecSum),
			(PrepSquare),
			(ExecProd),
		);

		let client = WgpuRuntime::client(&Default::default());
		let binding = score.1.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!("Output = {:?}", output_values);
		Ok(())
	}

	#[test]
	fn test_action_space_matrix() -> Result<()> {
		let client = WgpuRuntime::client(&Default::default());
		let meta = create_meta_matrix();
		let (meta_input, handle_input) = meta.handle_from_vec::<WgpuRuntime>(vec![
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
		]);
		let some_tensor = to_tref::<WgpuRuntime>((&meta_input, &handle_input));

		let score = action_space!(
			WgpuRuntime,
			(some_tensor, ExecMean, output_handle),
			(
				(&meta_input, &handle_input, &output_handle),
				PrepResiduals
			),
			(ExecSum),
			(PrepSquare),
			(ExecProd),
		);

		let binding = score.1.binding();
		let bytes = client.read_one(binding);
		let output_values = f32::from_bytes(&bytes);
		println!("Output = {:?}", output_values);
		Ok(())
	}
}
