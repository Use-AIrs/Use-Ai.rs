## [Calculator and Proc Macros](https://github.com/Use-AIrs/Use-Ai.rs/tree/main/crates/core/lib-calculator)
### Current
Here we find the first structures of how the Gpu Backends will be called, build and executed.

`lib.rs` gives us the first structure which needs to be provided by stage.

```rust
pub struct MetaData {
    pub stride: Box<[usize]>,
    pub shape: Box<[usize]>,
}
```

Every array needed for the operation needs to be provided by `lib-stage` as `MetaData` and its `Vec<f32>` representation.

- `pub fn build(stride: Box<[usize]>, shape: Box<[usize]>) -> MetaData {...}`: Can be used to build `MetaData`.
- `pub fn handle_empty<R: Runtime>(&self) -> (&Self, Handle) {...}` builds a tuple containing `&self` and a empty `Handle` of a gpu client.
- `pub fn handle_from_vec<R: Runtime>(&self, input: Vec<f32>) -> (&Self, Handle) {...}` builds a tuple containing `&self` and a filled `Handle` of a gpu client.

`model`'s `mod.rs` contains the `Operation` trait.

```rust
pub trait Operation: Sized {
    type Ctx;
    type Operator;
    type Output;

    fn exec(ctx: Self::Ctx, operator: Self::Operator) -> Self::Output;
}
```

Every `Operation` needs a `Ctx`, which can be implemented trough `#[ctx]` and an Operator which needs to be implemented
trough `#[operator]`. The Operator macro allow us to build a `TensorHandleRef` tuple out of `GbdtHandle`'s which will be
used to build the `Operation` on the GPUs Kernel.

The Operators tuple is our representation of the Memory in the GPU. For now, `#[operator]` just builds an implementation
over a structure like

```rust
pub struct GbdtOperator {
    pub target: (MetaData, Handle),
    pub table: (MetaData, Handle),
    pub buffer: (MetaData, Handle),
}
```

to build a tuple of `TensorHandleRef<R>` which will be lowered with [CubeCL](https://github.com/tracel-ai/cubecl) into
the Kernel with a specific `Operation` which contains the needed Kernel of the Model.

### Future
In the future we will lower all implementations of `lib-calculator` into the GPU Kernel with CubeCL. The goal here is that
we `lib-calculator` also can help with standardization in `lib-stage`.