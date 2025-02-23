#[cfg(test)]
mod tests {
    use crate::error::Result;
    use crate::operator::exec::base::PipelineExec;
    use crate::operator::exec::exec_argmax::ExecArgMax;
    use crate::operator::exec::exec_argmin::ExecArgMin;
    use crate::operator::exec::exec_mean::ExecMean;
    use crate::operator::exec::exec_prod::ExecProd;
    use crate::operator::exec::exec_sum::ExecSum;
    use cubecl::client::ComputeClient;
    use cubecl::prelude::{Runtime, TensorHandleRef};
    use cubecl::server::Handle;
    use cubecl::wgpu::WgpuRuntime;
    use cubecl::CubeElement;

    fn tensor_handle_ref(
        client: &ComputeClient<<WgpuRuntime as Runtime>::Server, <WgpuRuntime as Runtime>::Channel>,
    ) -> TensorHandleRef<'static, WgpuRuntime> {
        let handle = client.create(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
        let static_handle: &'static Handle = Box::leak(Box::new(handle));
        //unsafe { TensorHandleRef::from_raw_parts(static_handle, &[1,2], &[2, 2], 4) }
        unsafe { TensorHandleRef::from_raw_parts(static_handle, &[1, 1], &[1, 4], 4) }
    }

    #[test]
    fn test_exec_mean() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let dummy_tensor = tensor_handle_ref(&client);
        let output_handle = ExecMean::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        println!("Output = {:?}", output_values);
        Ok(())
    }

    #[test]
    fn test_exec_argmin() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let dummy_tensor = tensor_handle_ref(&client);
        let output_handle = ExecArgMin::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        println!("Output = {:?}", output_values);
        Ok(())
    }

    #[test]
    fn test_exec_argmax() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let dummy_tensor = tensor_handle_ref(&client);
        let output_handle = ExecArgMax::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        println!("Output = {:?}", output_values);
        Ok(())
    }

    #[test]
    fn test_exec_prod() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let dummy_tensor = tensor_handle_ref(&client);
        let output_handle = ExecProd::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        println!("Output = {:?}", output_values);
        Ok(())
    }

    #[test]
    fn test_exec_sum() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let dummy_tensor = tensor_handle_ref(&client);
        let output_handle = ExecSum::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        println!("Output = {:?}", output_values);
        Ok(())
    }
}
