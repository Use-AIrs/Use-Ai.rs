# Welcome to Use-AI.rs!

At **Use-AI.rs**, we are building an open-source AI framework in Rust.  
Our goal is to create a concurrent, locally hostable AI agent for practical applications in production environments.

To achieve this goal, we are building on top of [Burn](https://burn.dev/) and [CubeCL](https://github.com/tracel-ai/cubecl).  
We are working with a JSON configuration standard. This configuration file contains all the information required to transform data and execute various AI operations. CubeCL provides us with the necessary operational tools to achieve our aim of delivering a practical AI framework.

For production coding, we will focus on implementing reinforcement learning (RL) algorithms such as **Q-Learning** and **Deep Q-Networks**. Additionally, we plan to implement ensemble learning algorithms. In the first iteration, we will use **gradient-based decision trees (GBDT)**.

To provide insights into our decision-making process and document early-stage changes, the maintainer is maintaining a small blog in the [discussions section](https://github.com/Use-AIrs/use-ai.rs/discussions).


## Staging Tool

This tool represents the highest level of abstraction we provide, in the form of a **CLI tool**.  
The CLI tool will include:

- A **configuration manager** and
- A **staging tool** for exploring the features of Use-AI.rs with custom data.

### Features:
- **No prior Rust knowledge required**: Users can explore the tool without needing to understand Rust.
- **Comprehensive implementation guidance**: The tool's code includes detailed explanations for production use.
- **Prerequisites**: Rust and MongoDB must be installed.

This tool is built using [Inquire](https://docs.rs/inquire/latest/inquire/).


## Core Components

### Store

The **Store** component provides a simple synchronous [MangoDB](https://www.mongodb.com/) storage layer for:
- Storing configurations.
- Storing outputs.

The stored data can be downstreamed for further use.

### Transformer

The **Transformer** component processes and prepares abstract information for the Operator.  
Key features:
- Prepares data for downstream processing.
- Parallelizes some operations using [Rayon](https://docs.rs/rayon/latest/rayon/) on the CPU.

**Future Plans**:
- **High-Performance Computing (HPC)**: Enhance parallelism and optimization for large-scale data streams.

### Operator

The **Operator** component processes prepared data into **Burn Tensors**.  
Key features:
- Executes Tensors using the **WGPU backend**.
- Designed with reinforcement learning (RL) in mind.

**Future Plans**:
- Build custom GPU kernels using **CubeCL** for more complex RL operations.

##### Let us know your thoughts or contribute to the project!
