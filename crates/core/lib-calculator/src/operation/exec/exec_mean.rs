use crate::error::{CalcError, Result};
use crate::operation::exec::base::PipelineExec;
use cubecl::client::ComputeClient;
use cubecl::prelude::TensorHandleRef;
use cubecl::reduce::instructions::Mean;
use cubecl::reduce::reduce;
use cubecl::wgpu::WgpuRuntime;
use cubecl::Runtime;

pub struct ExecMean;

impl<R: Runtime> PipelineExec<R> for ExecMean {
    fn exec(
        op: TensorHandleRef<'_, R>,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<()> {
        if op.shape.len() == 1 {
            let output_handle = client.empty(4);
            let output = unsafe {
                TensorHandleRef::<'_, R>::from_raw_parts(&output_handle, &[1, 2], &[2, 1], 4)
            };
            let res = reduce::<R, f32, f32, Mean>(&client, op, output, 0, None);
            if res.is_ok() {
                Ok(())
            } else {
                Err(CalcError::GpuError)
            }
        } else if op.shape.len() == 2 {
            let factor = op.shape[0] * op.shape[1];
            let sh = op.shape[0];
            let st = op.strides[1];
            let shape = [sh, 1];
            let strides = [st, 1];

            let output_handle = client.empty(factor * 4);
            let output = unsafe {
                TensorHandleRef::<'_, R>::from_raw_parts(&output_handle, &strides, &shape, 4)
            };
            reduce::<R, f32, f32, Mean>(&client, op, output, 1, None)?;
            Ok(())
        } else {
            Err(CalcError::GpuError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gbdt::GbdtOperator;
    use crate::operation::Operator;
    use crate::MetaData;

    #[test]
    fn pipeline_mean() -> Result<()> {
        let client = WgpuRuntime::client(&Default::default());
        let md = MetaData::build(Box::new([2, 2]), Box::new([2, 1]));
        let (meta, handle) =
            MetaData::handle_from_vec::<WgpuRuntime>(&md, vec![1.0, 2.0, 3.0, 4.0]);
        let mdb = MetaData::build(Box::new([2, 2]), Box::new([2, 1]));
        let (metab, handleb) =
            MetaData::handle_from_vec::<WgpuRuntime>(&mdb, vec![5.0, 6.0, 7.0, 8.0]);
        let mdc = MetaData::build(Box::new([2, 2]), Box::new([2, 1]));
        let (metac, handlec) =
            MetaData::handle_from_vec::<WgpuRuntime>(&mdc, vec![5.0, 6.0, 7.0, 8.0]);

        let op = GbdtOperator {
            target: (meta.clone(), handle),
            table: (metab.clone(), handleb),
            buffer: (metac.clone(), handlec),
        };

        let (t0, t1, t2) = Operator::<WgpuRuntime>::tensor_refs(&op);

        ExecMean::exec(t0, &client)
    }
}
