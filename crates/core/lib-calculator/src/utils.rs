use crate::MetaData;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::Runtime;

/// This translates our tuples (&MetaData, Handle) to a TensoprHandleRef
pub fn to_tref<'a, R: Runtime>(input: (&'a MetaData, &'a Handle)) -> TensorHandleRef<'a, R> {
	unsafe {
		TensorHandleRef::<'a, R>::from_raw_parts(
			input.1,
			&*input.0.stride,
			&*input.0.shape,
			std::mem::size_of::<f32>(),
		)
	}
}
