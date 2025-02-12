//! For an Operation we first need to build an Operator.
//! A push takes some values in one form or another and builds an Operator with these.
//! When some primitive data is pushed/build into an Operator we can execute the Operator.
//! The OperationResult can the ether be finalized or can be pushed again to build more complex operations.
//! For now, we just implement everything with pushes on the Cpu. But our approach will allow us to
//! lower this mechanism completely into a Gpu kernel when introducing a counter and an allocator.

use crate::operation::exec::base::PipelineExec;
use crate::operation::push::base::PipelinePush;
use cubecl::channel::ComputeChannel;
use cubecl::prelude::*;
use cubecl::reduce::*;

pub mod error;
pub mod exec;
pub mod push;

pub trait Operator<R: Runtime> {
    type Tuple<'a>
    where
        Self: 'a;

    fn tensor_refs<'a>(&'a self) -> Self::Tuple<'a>;
}

pub trait Context {
    type Tuple;

    fn ctx_ref(&self) -> Self::Tuple;
}
