use std::marker::PhantomData;
use cubecl::ir::{ConstantScalarValue, UIntKind};
use cubecl::ir::VariableKind::ConstantScalar;
use crate::error::{CalcError, Result};
use crate::operator::push::base::PipelinePush;
use crate::MetaData;
use cubecl::prelude::*;
use cubecl::server::Handle;
/*
pub struct PrepResiduals<R: Runtime> {
    _res: PhantomData<R>,
}

impl<R: Runtime> PipelinePush<R> for PrepResiduals<R> {
    type Input<'i> = (&'i MetaData, &'i Handle, &'i Handle);
    type Output = (MetaData, Handle);

    fn push<'i>(
        input: Self::Input<'i>,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(MetaData, Handle)> {
        let (meta, tensor_handle, scalar_handle) = input;
        let input_tensor = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                tensor_handle,
                &meta.stride,
                &meta.shape,
                4,
            )
        };
        println!("{:?}", scalar_handle);

        if meta.stride.to_vec() == &[1, 1] {
            let n = meta.shape[1];
            let output_shape = [2, n];
            let output_strides = [n, 1];
            let total_bytes = 2 * n * 4;
            let output_handle = client.empty(total_bytes);
            let mut output_tensor = unsafe {
                TensorHandleRef::<R>::from_raw_parts(
                    &output_handle,
                    &output_strides,
                    &output_shape,
                    4
                )
            };

            let binding = scalar_handle.clone().binding();
            let bytes = client.read_one(binding);
            let output_value = f32::from_bytes(&bytes);
            let value = output_value[0];
            println!("{:?}", output_value);
            let scalar_arg = ScalarArg::<f32>::new(value);
            let cubecount = CubeCount::new_2d(1u32, 1u32);
            let cubedim = CubeDim::new(n as u32, 2u32,1u32);
            unsafe {
                prep_residuals::launch_unchecked::<f32, R>(
                    &client,
                    cubecount,
                    cubedim,
                    output_tensor.as_tensor_arg(1),
                    input_tensor.as_tensor_arg(1),
                    scalar_arg,
                );
            }

            let meta = MetaData::build(Box::new(output_strides), Box::new(output_shape));
            Ok((meta, output_handle))
        } else {
            let m = meta.shape[0];
            let n = meta.shape[1];
            let output_shape = [m, n, 2];
            let output_strides = [n * 2, 2, 1];
            let total_bytes = m * n * 2 * 4;
            let output_handle = client.empty(total_bytes);
            let mut output_tensor = unsafe {
                TensorHandleRef::<R>::from_raw_parts(
                    &output_handle,
                    &output_strides,
                    &output_shape,
                    4
                )
            };

            let binding = scalar_handle.clone().binding();
            let bytes = client.read_one(binding);
            let output_value = f32::from_bytes(&bytes);
            let value = output_value[0];
            let scalar_arg = ScalarArg::<f32>::new(-value);
            let cubecount = CubeCount::new_3d(1u32,  1u32);
            let cubedim = CubeDim::new(m as u32, n as u32,2u32);
            println!("{:?}", cubedim);

            unsafe {
                prep_residuals::launch_unchecked::<f32, R>(
                    &client,
                    cubecount,
                    cubedim,
                    output_tensor.as_tensor_arg(1),
                    input_tensor.as_tensor_arg(1),
                    scalar_arg,
                );
            }
            let meta = MetaData::build(Box::new(output_strides), Box::new(output_shape));
            Ok((meta, output_handle))
        }
    }
}
#[cube(launch_unchecked)]
pub fn prep_residuals<T: Numeric>(
    input: &Tensor<T>,
    output: &mut Tensor<T>,
    scalar: T,
){
    let len = input.len();
    let tensor = [scalar; len.constant(];
    let len = output.len();
    let half = len / 2;
    for i in 0u32..len {
        if i < half {
            output[i] = input[i];
        } else {
            output[i] = scalar.copy();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetaData;
    use cubecl::client::ComputeClient;
    use cubecl::wgpu::WgpuRuntime;
    use cubecl::prelude::TensorHandleRef;
    use crate::error::Result;

    fn create_meta_vector() -> MetaData {
        MetaData::build(Box::new([1, 1]), Box::new([1, 6]))
    }

    fn create_meta_matrix() -> MetaData {
        MetaData::build(Box::new([6, 1]), Box::new([2, 6]))
    }

    #[test]
    fn test_prep_residuals_vector() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let meta = create_meta_vector();
        let (meta, handle) = meta.handle_from_vec::<WgpuRuntime>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let scalar_handle = client.create(f32::as_bytes(&[3.0]));

        let (output_meta, output_handle) =
            PrepResiduals::<WgpuRuntime>::push((meta, &handle, &scalar_handle), &client)?;

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
        let (meta, handle) = meta.handle_from_vec::<WgpuRuntime>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let scalar_handle = client.create(f32::as_bytes(&[3.0]));

        let (output_meta, output_handle) =
            PrepResiduals::<WgpuRuntime>::push((meta, &handle, &scalar_handle), &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        println!("Output = {:?}", output_values);
        Ok(())
    }
}

 */