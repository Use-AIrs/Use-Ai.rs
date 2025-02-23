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
        unsafe { TensorHandleRef::from_raw_parts(static_handle, stride, shape, data.len()) }
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
        unsafe { TensorHandleRef::from_raw_parts(static_handle, stride, shape, total) }
    }

    #[test]
    fn test_exec_mean_vector() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let len = 1000;
        let shape = Box::leak(Box::new([1, len]));
        let stride = Box::leak(Box::new([1, 1]));
        let dummy_tensor = create_random_vector(&client, len, shape, stride);
        let start = Instant::now();
        let output_handle = ExecMean::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecMean (Vector) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_argmin_vector() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let len = 1000;
        let shape = Box::leak(Box::new([1, len]));
        let stride = Box::leak(Box::new([1, 1]));
        let dummy_tensor = create_random_vector(&client, len, shape, stride);
        let start = Instant::now();
        let output_handle = ExecArgMin::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecArgMin (Vector) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_argmax_vector() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let len = 1000;
        let shape = Box::leak(Box::new([1, len]));
        let stride = Box::leak(Box::new([1, 1]));
        let dummy_tensor = create_random_vector(&client, len, shape, stride);
        let start = Instant::now();
        let output_handle = ExecArgMax::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecArgMax (Vector) Output: {:?}, Time: {:?}  Undefined behavior",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_prod_vector() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let len = 100000000;
        let shape = Box::leak(Box::new([1, len]));
        let stride = Box::leak(Box::new([1, 1]));
        let dummy_tensor = create_random_vector(&client, len, shape, stride);
        let start = Instant::now();
        let output_handle = ExecProd::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecProd (Vector) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_sum_vector() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let len = 1000;
        let shape = Box::leak(Box::new([1, len]));
        let stride = Box::leak(Box::new([1, 1]));
        let dummy_tensor = create_random_vector(&client, len, shape, stride);
        let start = Instant::now();
        let output_handle = ExecSum::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecSum (Vector) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_mean_matrix() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let shape = Box::leak(Box::new([25, 40]));
        let stride = Box::leak(Box::new([40, 1]));
        let dummy_tensor = create_random_matrix(&client, shape, stride);
        let start = Instant::now();
        let output_handle = ExecMean::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecMean (Matrix) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_argmin_matrix() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let shape = Box::leak(Box::new([25, 40]));
        let stride = Box::leak(Box::new([40, 1]));
        let dummy_tensor = create_random_matrix(&client, shape, stride);
        let start = Instant::now();
        let output_handle = ExecArgMin::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecArgMin (Matrix) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_argmax_matrix() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let shape = Box::leak(Box::new([25, 40]));
        let stride = Box::leak(Box::new([40, 1]));
        let dummy_tensor = create_random_matrix(&client, shape, stride);
        let start = Instant::now();
        let output_handle = ExecArgMax::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecArgMax (Matrix) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_prod_matrix() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let shape = Box::leak(Box::new([25, 40]));
        let stride = Box::leak(Box::new([40, 1]));
        let dummy_tensor = create_random_matrix(&client, shape, stride);
        let start = Instant::now();
        let output_handle = ExecProd::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecProd (Matrix) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }

    #[test]
    fn test_exec_sum_matrix() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let shape = Box::leak(Box::new([25, 40]));
        let stride = Box::leak(Box::new([40, 1]));
        let dummy_tensor = create_random_matrix(&client, shape, stride);
        let start = Instant::now();
        let output_handle = ExecSum::exec(dummy_tensor, &client)?;
        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = f32::from_bytes(&bytes);
        let elapsed = start.elapsed();
        println!(
            "ExecSum (Matrix) Output: {:?}, Time: {:?}",
            output_values, elapsed
        );
        Ok(())
    }
}