use crate::error::Result;
use crate::model::gbdt::{DataSet, GbdtOperator, GbdtRules, XGBoostModel};
use crate::model::Operation;
use crate::operator::Operator;
use cubecl::prelude::*;
use cubecl::reduce::*;

pub struct GbdtTrainer;

impl<R: Runtime> Operation<R> for GbdtTrainer {
    type Ctx = GbdtRules;
    type Op = GbdtOperator;

    fn exec(ctx: Self::Ctx, operator: Self::Op, client: ComputeClient<R::Server, R::Channel>) {
        let mem: (TensorHandleRef<R>, TensorHandleRef<R>, TensorHandleRef<R>) = operator.mem_rep();
    }
}

impl XGBoostModel {}
