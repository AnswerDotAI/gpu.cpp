import struct

import numpy as np

import gpu_cpp as gpu


scale = r'''
struct Params { factor: f32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&input)) {
    output[id.x] = input[id.x] * params.factor;
  }
}
'''


# Upload NumPy data, bind access explicitly, dispatch, and recover its shape and dtype.
context = gpu.Context()
values = np.arange(12, dtype=np.float32)
gpu_values = gpu.tensor(context, values)
gpu_output = gpu.create_tensor(context, values.shape, gpu.f32)
shader = gpu.WGSL(scale, workgroup_size=[4, 1, 1])
bindings = [gpu.read(gpu_values), gpu.read_write(gpu_output)]
kernel = gpu.create_kernel(context, shader, bindings, workgroups=[3, 1, 1], parameters=struct.pack('<f', 3))

dispatched = gpu.dispatch_kernel(context, kernel)
gpu.wait(context, dispatched)
np.testing.assert_array_equal(gpu.to_numpy(context, gpu_output), values * 3)
assert gpu_output.shape == [12]
assert gpu_output.dtype == gpu.f32

# The tensor and compiled kernel remain reusable with new input data.
updated = np.arange(12, dtype=np.float32)[::-1].copy()
gpu.to_gpu(context, gpu_values, updated)
dispatched = gpu.dispatch_kernel(context, kernel)
gpu.wait(context, dispatched)
np.testing.assert_array_equal(gpu.to_numpy(context, gpu_output), updated * 3)

print('Python upload, reusable dispatch, and NumPy readback story passed')
