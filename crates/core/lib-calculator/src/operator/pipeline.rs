use cubecl::Runtime;

pub struct OperatorChain<R: Runtime> {
    mem: OperatorMem<R>,
}