# New Blog and Doc Updates

## Maintainer's Comment | 03.02.2025

### Welcome to the new Blog

As you may have noticed the new Blogs place will also will be used as entry place for the *Use-AI.rs* Documentation. This
was decided due to the fact, that it's easier to maintain a good documentation within the codebase compared to an external
Application. So, **Welcome to the new Documentation!** This blog will be useful to communicate changes and discuss
development decisions for future users of the Framework.

### State

Today I will go tough the hole Project and will discuss the decisions made. This will be structured from the highest abstraction
used in Use-AI.rs down to the lowest abstraction layers. This represents the architecture chosen.
These discussions will also be used in the Book to introduce the idea of each component.

### [Use-AI](https://github.com/Use-AIrs/Use-Ai.rs/tree/main/crates/use-ai)
#### Current
*Use-AI* is just a basic cli tool written with [Inquire](https://github.com/mikaelmello/inquire). Inquire is used since
we just want to have a nice little entry point for People who doesn't now the project and want to find out what with data
can be done with *Use-AI.rs*. Also, Inquire provides an interface which allows you to easily find the underlying functions 
so you have an example of how Use-AI.rs can be used in production. 

#### Future
The Use-AI Tool is just the first tool we want to provide. In Future the goal is to provide a hole Network Server Layer.
For the Server layer we are thinking about a TCP Server. The Server should provide a load test tool for a hypothetical 
Query Server. But since this is a plan in far future these are only idea's. When someone I found who can implement something
like that it may be implemented in parallel.

### [Store](https://github.com/Use-AIrs/Use-Ai.rs/tree/main/crates/core/lib-store)
#### Current
`lib-store` is the first library of the Core of *Use-AI.es*. Here we find `ai_config.rs` and `mangodb.rs`.

`ai_config.rs`, is the entrypoint for the configuration file.

  - `Models`, will be provided by `lib-calculator` and
  - `DataSection`, will be provided by `lib-stage`.

`mangodb.rs`, operates [MangoDB](https://www.mongodb.com/docs/drivers/rust/current/) in synchronous. 
Since we expect this Db to run on a system what also handles our data in parallel on the cpu and 
since we don't have to worry about huge db requests it is decided that we only communicate 
synchronised with the Db so we don't spawn threads that block building threads of performance critical features.

#### Future
Not much will happen here. Only when the Config is expanded we will see some changes here. Some additional Db request
functions and some functions regarding Configs will be added here when needed.

### [Stage](https://github.com/Use-AIrs/Use-Ai.rs/tree/main/crates/core/lib-stage)
#### Current
Here is where the construction site begins. Currently, we only have a CSV pipeline to show what the plan is with `lib-stager`.
This pipeline shows how we want to handle data on a more abstract layer. For now, we just input some csv file and handle 
it from there with different operations on columns or the hole data. This is chosen because we want to reach a first complete
pass through *Use-AI.rs* using a simple XGBoost algorithm for now. Till we expanded that simple XGBoost it's called GBDT here.

#### Future
For DQNs and Q-learning we need to expand `lib-stage`. Goal is not to only use files instead this will be used to Configure
a Data Stream which first, will enable the possibility to get Input from program external drivers and also, will add the 
possibility for communication between models. But to be able to implement such features we first need to focus on the 
`lib-calculator`.
The functions as they are, will be renamed in near Future.

### [Calculator and Proc Macros](https://github.com/Use-AIrs/Use-Ai.rs/tree/main/crates/core/lib-calculator)
#### Current
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

#### Future
In the future we will lower all implementations of `lib-calculator` into the GPU Kernel with CubeCL. The goal here is that
we `lib-calculator` also can help with standardization in `lib-stage`.