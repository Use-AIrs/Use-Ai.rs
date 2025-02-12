use crate::error::Result;
use crate::model::gbdt::{GbdtOperator, GbdtRules};
use crate::model::Operation;
use cubecl::prelude::*;
use cubecl::reduce::*;
pub struct GbdtTrainer;

impl<R: Runtime> Operation<R> for GbdtTrainer {
    type Ctx = GbdtRules;
    type Op = GbdtOperator;
    type Output = Result<()>;

    fn exec(
        ctx: Self::Ctx,
        operator: Self::Op,
        client: ComputeClient<R::Server, R::Channel>,
    ) -> Self::Output {
        todo!()
    }
}
