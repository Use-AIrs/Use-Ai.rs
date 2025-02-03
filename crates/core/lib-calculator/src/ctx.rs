//! Here we find the bridge between Cpu and Gpu. The Cpu provides a CpuCtx defined here.
//! With the CpuCtx we can build the GpuCtx. With the GpuCtx's content we can build an operation.

/// This struct needs to be provided by the CPU through lib-stage.
pub struct CpuCtx {
    pub input: Vec<f32>,
    pub input_stride: Box<[usize]>,
    pub input_shape: Box<[usize]>,

    pub output_stride: Box<[usize]>,
    pub output_shape: Box<[usize]>,

    pub model: Model,
}

pub enum Model {
    GBDT(Mode),
}

pub enum Mode {
    Training,
    Predict,
}
