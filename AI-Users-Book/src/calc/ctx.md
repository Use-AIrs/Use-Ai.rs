# Context

## CpuCtx:

This structure defines 

```rust
pub struct CpuCtx {
    pub input: ArrayD<f32>,
    pub output: Option<ArrayD<f32>>,

    pub output_stride: Box<[usize]>,
    pub output_shape: Box<[usize]>,
    pub output_size: usize,
    pub model: Model,
    pub mode: Mode,
}
```