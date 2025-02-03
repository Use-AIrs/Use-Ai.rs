# Calculator

The *Calculator* is devided into two layers. First we got a communication layer for cpu and gpu.
In the communication layer we find a [*Config*](config.md) and a [*Context*](ctx.md). This provides the nesesarry Structures 
to deploy an operation from the CPU. In Future [*Operation*](operation.md) will change to gpu_operation. 
From that moment on every calculation operation is deployed on the gpu.