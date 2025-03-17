pub mod config;
mod error;
mod model;
mod operator;
mod utils;

use cubecl::server::Handle;
use cubecl::{CubeElement, Runtime};

#[derive(Debug, Clone)]
pub struct CpuTensor {
	pub handle: Box<Handle>,
	pub meta: MetaData,
}

impl CpuTensor {
	fn build(
		handle: Box<Handle>,
		stride: &[usize],
		shape: &[usize],
	) -> Self {
		let meta = MetaData::build(stride.into(), shape.into());
		Self { handle, meta }
	}
}

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
	pub fn cputensor_empty<R: Runtime>(self) -> CpuTensor {
		let client = R::client(&Default::default());
		let size_f32 = std::mem::size_of::<f32>();
		let size = &self.shape.iter().product::<usize>() * size_f32;
		let handle = client.empty(size);
		CpuTensor {
			handle: Box::new(handle),
			meta: self,
		}
	}
	pub fn cputensor_from_vec<R: Runtime>(
		self,
		input: Vec<f32>,
	) -> CpuTensor {
		let client = R::client(&Default::default());
		let put = input.as_slice();
		let handle = client.create(f32::as_bytes(put));
		CpuTensor {
			handle: Box::new(handle),
			meta: self,
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
		let cpu_tensor = MetaData::cputensor_empty::<WgpuRuntime>(md);
		let len = cpu_tensor.handle.size();
		assert_eq!(len, 4);
	}
	#[test]
	fn md_test_vec() {
		let md = MetaData::build(Box::new([2, 2]), Box::new([2, 1]));
		let cpu_tensor = MetaData::cputensor_from_vec::<WgpuRuntime>(md, vec![1.0, 2.0, 3.0, 4.0]);
		let len = cpu_tensor.handle.size();
		assert_eq!(len, 16);
	}
}
