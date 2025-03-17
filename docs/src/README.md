At Use-AI.rs, we are building an open-source AI framework in Rust.
Our goal is to create a concurrent, locally hostable AI agent for practical applications in production environments.

The different layers of abstraction build spaces for different professions. For **Business Informatic Specialists**, **Mathematicians**, and **Computer Scientists**, Use-AI.rs provides a comprehensive set of tools and libraries to build and extend AI applications.

The entrypoint of Use-AI.rs is the [`use_ai`](/tooling/useai.md) crate. It provides a toolset for **Business Informatic Specialists** and **Mathematicians** to test and deploy AI applications with ease.
[`use_ai`](/tooling/useai.md) also is a good starting point for **Computer Scientists** since it shows how to implement the underlying architecture of the framework.

The underlying architecture is mostly interesting for **Computer Scientists** and can be found in the `core` crate. It contains the following modules:
- [`lib-store`](/store/store.md): A synchronous storage module with [MangoDB](https://www.mongodb.com/).
- [`lib-stage`](/stage/stage.md): Handles data processing and normalization with [Rayon](https://github.com/rayon-rs/rayon) and [ndArray](https://github.com/rust-ndarray/ndarray).
- [`lib-calculator`](/calc/calc.md): Here we operate the GPU and build AI models with [CubeCL](https://github.com/tracel-ai/cubecl).
- [`lib-proc_macros`](/macros/macros.md): A set of procedural macros for simplifying common tasks in AI development.

At [`lib-calculator::model`](/calc/model.md) we find a space also interesting for **Mathematicians** since it provides a set of easy to use tensor operations that can be used to build powerful AI models.
