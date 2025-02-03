//! For an Operation we first need to build an Operator.
//! A push takes some values in one form or another and builds an Operator with these.
//! When some primitive data is pushed/build into an Operator we can execute the Operator.
//! The OperationResult can the ether be finalized or can be pushed again to build more complex operations.
//! For now, we just implement everything with pushes on the Cpu. But our approach will allow us to
//! lower this mechanism completely into a Gpu kernel when introducing a counter and an allocator.

use crate::operation::gpu_ctx::GpuCtx;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::Runtime;
use std::process::Output;

mod exec;
mod gpu_ctx;
mod push;

pub struct Operator<R: Runtime> {
    pub client: ComputeClient<R::Server, R::Channel>,
    pub t0: Option<Memory>,
    pub t1: Option<Memory>,
    pub t2: Option<Memory>,
}

impl<R: Runtime> Operator<R> {
    pub fn new(gpu_ctx: GpuCtx<R>) -> Self {
        Operator {
            client: gpu_ctx.client,
            t0: None,
            t1: None,
            t2: None,
        }
    }
}

pub struct Memory {
    pub handle: Handle,
    pub strides: Box<[usize]>,
    pub shape: Box<[usize]>,
}

pub struct OperatorResult<R: Runtime> {
    pub op: Operator<R>,
    pub output: OperationResult,
}

/// This is just for the CPU times to simplify. When we handle this on the GPU we will no longer be
/// restricted to any dimension.
#[derive(Debug)]
pub enum OperationResult {
    Value(f32),
    Arr1(Vec<f32>),
    Arr2(Vec<Vec<f32>>),
}
