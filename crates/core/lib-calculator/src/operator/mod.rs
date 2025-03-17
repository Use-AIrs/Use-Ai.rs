//! For an Operation we first need to build an Operator.
//! A push takes some values in one form or another and builds an Operator with these.
//! When some primitive data is pushed/build into an Operator we can execute the Operator.
//! The OperationResult can the ether be finalized or can be pushed again to build more complex operations.
//! For now, we just implement everything with pushes on the Cpu. But our approach will allow us to
//! lower this mechanism completely into a Gpu kernel when introducing a counter and an allocator.

pub use crate::operator::exec::*;
pub use crate::operator::push::*;

use cubecl::prelude::*;

pub mod error;
pub mod exec;
pub mod push;
mod test;

pub trait Operator<R: Runtime> {
	type Mem<'a>
	where
		Self: 'a;

	fn mem_rep<'a>(&'a self) -> Self::Mem<'a>;
}

pub trait Context {
	type Tuple;

	fn ctx_ref(&self) -> Self::Tuple;
}
