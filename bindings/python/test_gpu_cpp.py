import gpu_cpp as gpu
import numpy as np

ctx = gpu.create_context()

N = 12

input = gpu.create_tensor(ctx, [N], gpu.kf32)
output = gpu.create_tensor(ctx, [N], gpu.kf32)
kernel_code = gpu.create_kernel_code(
    """
    const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
    @group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
    @group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
    @group(0) @binding(1) var<storage, read_write> dummy: array<{{precision}}>;
    @compute @workgroup_size({{workgroupSize}})
    fn main(
        @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
        let i: u32 = GlobalInvocationID.x;
        if (i < arrayLength(&inp)) {
            let x: f32 = inp[i];
            out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                     * (x + .044715 * x * x * x))), x, x > 10.0);
        }
    }
    """,
    256,
    gpu.kf32
    )

kernel = gpu.create_kernel(ctx, kernel_code, [input, output], [0,0], [12,1,1])

gpu.to_gpu_float(ctx, np.array([1,2,3,4,1,2,3,4,1,2,3,4],np.float32), input)

gpu_async = gpu.dispatch_kernel(ctx, kernel);

gpu.wait(ctx, gpu_async);

print(gpu.to_cpu_float(ctx, output))
