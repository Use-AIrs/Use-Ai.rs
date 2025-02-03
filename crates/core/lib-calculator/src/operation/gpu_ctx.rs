use crate::ctx::CpuCtx;
use crate::error::{CalcError, Result};
use cubecl::prelude::*;
use cubecl::server::Handle;

#[derive(Debug)]
pub struct GpuCtx<R: Runtime> {
    pub client: ComputeClient<R::Server, R::Channel>,
    pub size_f32: usize,

    pub input_handle: Handle,
    pub input_shape: Box<[usize]>,
    pub input_strides: Box<[usize]>,

    pub output_shape: Box<[usize]>,
    pub output_strides: Box<[usize]>,
}

impl<R: Runtime> GpuCtx<R> {
    pub fn build(cpu: CpuCtx, device: &R::Device) -> Result<Self> {
        let client = R::client(device);
        let size_f32 = size_of::<f32>();

        let input = cpu.input;
        let slice = input.as_slice();
        let handle = client.create(f32::as_bytes(slice));
        Ok(GpuCtx {
            client,
            size_f32,
            input_handle: handle,
            input_shape: cpu.input_shape,
            input_strides: cpu.input_stride,
            output_shape: cpu.output_shape,
            output_strides: cpu.output_stride,
        })
    }
}

#[cfg(test)]
mod gpu_ctx_tests {
    use crate::ctx::{CpuCtx, Mode, Model};
    use crate::error::CalcError::GpuError;
    use crate::error::{CalcError, Result};
    use crate::operation::gpu_ctx::GpuCtx;
    use cubecl::wgpu::WgpuDevice::DefaultDevice;
    use std::fmt::{Debug, Pointer};

    #[test]
    fn test_build_gpu() -> Result<()> {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cpu = CpuCtx {
            input,
            input_stride: Box::new([1, 3]),
            input_shape: Box::new([2, 4]),
            output_stride: Box::new([2, 2]),
            output_shape: Box::new([2, 1]),
            model: Model::GBDT(Mode::Training),
        };
        let gpu = GpuCtx::<cubecl::wgpu::WgpuRuntime>::build(cpu, &DefaultDevice);

        match gpu {
            Ok(ctx) => Ok(()),
            Err(e) => Err(GpuError),
        }
    }
}
