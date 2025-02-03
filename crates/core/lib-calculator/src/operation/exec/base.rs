pub trait PipelineExec {
    type Output;

    fn exec(self) -> Self::Output;
    fn finish(self) -> Self::Output;
}
