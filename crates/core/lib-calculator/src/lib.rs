mod error;
mod model;
mod operator;
mod utils;

use cubecl::prelude::TensorHandleRef;
use cubecl::{CubeElement, Runtime};

#[derive(Debug, Clone)]
pub struct MetaData {
	pub stride: Box<[usize]>,
	pub shape: Box<[usize]>,
}

impl MetaData {
	pub fn build(
		stride: Box<[usize]>,
		shape: Box<[usize]>,
	) -> Self {
		Self { stride, shape }
	}

	pub fn single() -> Self {
		Self {
			stride: Box::new([1, 1]),
			shape: Box::new([1, 1]),
		}
	}

	pub fn tensorref_empty<'o, R: Runtime>(self) -> TensorHandleRef<'o, R> {
		let client = R::client(&Default::default());
		let size_f32 = std::mem::size_of::<f32>();
		let size = &self.shape.iter().product::<usize>() * size_f32;
		let handle = Box::leak(Box::new(client.empty(size)));
		unsafe {
			TensorHandleRef::from_raw_parts(
				handle,
				Box::leak(self.stride),
				Box::leak(self.shape),
				size_f32,
			)
		}
	}

	pub fn tensorref_from_vec<'o, R: Runtime>(
		self,
		input: Vec<f32>,
	) -> TensorHandleRef<'o, R> {
		let put = input.as_slice();
		let client = R::client(&Default::default());
		let handle = Box::leak(Box::new(
			client.create(f32::as_bytes(put)),
		));
		unsafe {
			TensorHandleRef::from_raw_parts(
				handle,
				Box::leak(self.stride),
				Box::leak(self.shape),
				4,
			)
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use cubecl::wgpu::WgpuRuntime;
	#[test]
	fn md_test_empty() {
		let md = MetaData::build(Box::new([1]), Box::new([1]));
		let cpu_tensor = MetaData::tensorref_empty::<WgpuRuntime>(md);
		let len = cpu_tensor.handle.size();
		assert_eq!(len, 4);
	}
	#[test]
	fn md_test_vec() {
		let md = MetaData::build(Box::new([2, 2]), Box::new([2, 1]));
		let cpu_tensor = MetaData::tensorref_from_vec::<WgpuRuntime>(md, vec![1.0, 2.0, 3.0, 4.0]);
		let len = cpu_tensor.handle.size();
		assert_eq!(len, 16);
	}
}
