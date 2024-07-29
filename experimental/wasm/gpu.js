// gpu.js

class Shape {
    static kMaxRank = 8;
    
    constructor(...dims) {
        if (dims.length > Shape.kMaxRank) {
            throw new Error(`Shape can have at most ${Shape.kMaxRank} dimensions`);
        }

        this.rank = dims.length;
        
        // Initialize data with the provided dimensions
        this.data = dims;
        
        // Pad with zeros if necessary
        while (this.data.length < Shape.kMaxRank) {
            this.data.push(0);
        }
    }
}
class Array {
    constructor(buffer, usage, size) {
        this.buffer = buffer;
        this.usage = usage;
        this.size = size;
    }
}

class Tensor {
    constructor(data, shape) {
        this.data = data;  // This is now an Array instance
        this.shape = shape;
    }
}

async function createTensor(ctx, shape, dtype) {
    const numElements = size(shape);
    const byteSize = sizeBytes(dtype) * numElements;
    console.log("Creating tensor with shape", shape, "and dtype", dtype);
    console.log("size value is", byteSize);
    if (byteSize === 0) {
      throw new Error(`Cannot create a tensor with zero size. Shape: ${toString(shape)}, Type: ${dtype}`);
    }

    const buffer = ctx.device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    const arrayData = new Array(buffer, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, byteSize);
    const tensor = new Tensor(arrayData, shape);
    ctx.pool.data.set(buffer, tensor);
    return tensor;
}

class TensorView {
    constructor(data, offset = 0, span = data.data.size) {
        this.data = data;
        this.offset = offset;
        this.span = span;
    }
}

class Bindings {
    constructor(...args) {
        this.data = args.map(arg => arg instanceof TensorView ? arg.data : arg);
        this.viewOffsets = args.map(arg => arg instanceof TensorView ? arg.offset : 0);
        this.viewSpans = args.map(arg => arg instanceof TensorView ? arg.span : arg.data.size);
    }
}

class Context {
    constructor() {
        this.instance = null;
        this.adapter = null;
        this.device = null;
        this.queue = null;
        this.pool = new TensorPool(this);
        this.kernelPool = new KernelPool(this);
    }
}

class TensorPool {
    constructor(ctx) {
        this.ctx = ctx;
        this.data = new Map();
    }
}

class KernelPool {
    constructor(ctx) {
        this.ctx = ctx;
        this.data = new Set();
    }
}

class KernelCode {
    constructor(data, workgroupSize = new Shape(256, 1, 1), precision = NumType.kf32) {
        this.data = data;
        this.workgroupSize = workgroupSize;
        this.precision = precision;
        this.label = "kernel";
        this.entryPoint = "main";

        if (precision === NumType.kf16) {
            this.data = "enable f16;\n" + this.data;
        }
        this.data = replaceAll(this.data, "{{workgroupSize}}", toString(workgroupSize));
        this.data = replaceAll(this.data, "{{precision}}", toString(precision));
    }
}

class Kernel {
    constructor(device, bindGroup, computePipeline, nWorkgroups) {
        this.device = device;
        this.bindGroup = bindGroup;
        this.computePipeline = computePipeline;
        this.nWorkgroups = nWorkgroups;
        this.commandBuffer = null;
    }
}

// Enums and constants
const NumType = {
    kf16: 'f16',
    kf32: 'f32'
};

// Utility functions
function size(shape) {
    return shape.data.slice(0, shape.rank).reduce((a, b) => a * b, 1);
}

function sizeBytes(type) {
    switch (type) {
        case NumType.kf16: return 2;
        case NumType.kf32: return 4;
        default: throw new Error("Invalid NumType in size calculation.");
    }
}

function toString(obj) {
    if (obj instanceof Shape) {
        return obj.data.slice(0, obj.rank).join(', ');
    } else if (typeof obj === 'number') {
        return obj.toString();
    } else if (typeof obj === 'string') {
        return obj;
    } else if (obj && obj.constructor === Object) {
        // Handle plain objects (like NumType)
        for (let key in obj) {
            if (obj[key] === obj) return key;
        }
    }
    
    console.error("Unsupported type in toString:", obj);
    return String(obj);
}

function replaceAll(str, from, to) {
    return str.split(from).join(to);
}

function cdiv(n, d) {
    return Math.floor((n + d - 1) / d);
}

function cdivShape(total, group) {
    const result = new Shape();
    result.rank = total.rank;
    for (let dim = 0; dim < total.rank; ++dim) {
        result.data[dim] = cdiv(total.data[dim], group.data[dim]);
    }
    return result;
}

// Main functions
async function createContext() {
    const context = new Context();
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }
    context.instance = navigator.gpu;
    context.adapter = await context.instance.requestAdapter();
    if (!context.adapter) {
        throw new Error("Couldn't request WebGPU adapter.");
    }
    context.device = await context.adapter.requestDevice();
    context.queue = context.device.queue;
    return context;
}

function destroyContext(ctx) {
    console.log("Destroying context");
    ctx.queue = null;
    ctx.device = null;
    ctx.adapter = null;
    ctx.instance = null;
    console.log("Context destroyed");
}

function resetCommandBuffer(device, kernel) {
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(kernel.computePipeline);
    passEncoder.setBindGroup(0, kernel.bindGroup);
    passEncoder.dispatchWorkgroups(
        kernel.nWorkgroups.data[0],
        kernel.nWorkgroups.data[1],
        kernel.nWorkgroups.data[2]
    );
    passEncoder.end();
    kernel.commandBuffer = commandEncoder.finish();
}

async function createKernel(ctx, code, dataBindings, nWorkgroups, params = null) {
    const device = ctx.device;
    const numBindings = dataBindings.data.length + (params ? 1 : 0);
    
    const entries = dataBindings.data.map((tensor, i) => ({
        binding: i,
        resource: { buffer: tensor.data.buffer, size: tensor.data.size }
    }));

    if (params) {
        const uniformBuffer = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, params);
        entries.push({
            binding: numBindings - 1,
            resource: { buffer: uniformBuffer, size: params.byteLength}
        });
    }

    const bindGroupLayout = device.createBindGroupLayout({
        entries: entries.map((entry, i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { 
                type: i === numBindings - 1 && params ? 'uniform' : 'storage',
                hasDynamicOffset: false,
                minBindingSize: 0
            }
        }))
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: entries
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        code: code.data
    });

    const computePipeline = await device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: code.entryPoint
        }
    });

    const kernel = new Kernel(device, bindGroup, computePipeline, nWorkgroups);
    resetCommandBuffer(device, kernel);
    ctx.kernelPool.data.add(kernel);
    return kernel;
}

function toGPU(ctx, data, tensor) {
    ctx.device.queue.writeBuffer(tensor.data.buffer, 0, data);
}

async function toCPU(ctx, tensor, data) {
    const readBuffer = ctx.device.createBuffer({
        size: tensor.data.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = ctx.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(tensor.data.buffer, 0, readBuffer, 0, tensor.data.size);
    ctx.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const copiedData = readBuffer.getMappedRange();
    data.set(new Float32Array(copiedData));
    readBuffer.unmap();
    readBuffer.destroy();
}

function dispatchKernel(ctx, kernel) {
    ctx.device.queue.submit([kernel.commandBuffer]);
    return ctx.device.queue.onSubmittedWorkDone();
}

async function main() {
    console.log("Starting main");
    const ctx = await createContext();

    const inputTensor = await createTensor(ctx, new Shape(1024), NumType.kf32);
    const outputTensor = await createTensor(ctx, new Shape(1024), NumType.kf32);
    
    const inputData = new Float32Array(1024).fill(1.0);
    // const inputData = new Float32Array(1024);
    toGPU(ctx, inputData, inputTensor);
    
    const kernelCode = new KernelCode(`
        @group(0) @binding(0) var<storage, read_write> input_buffer: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let index = global_id.x;
            let input = input_buffer[index];
            output_buffer[index] = input * 2.0;
        }
    `);
      
    const kernel = await createKernel(ctx, kernelCode, new Bindings(inputTensor, outputTensor), new Shape(4, 1, 1));
    await dispatchKernel(ctx, kernel);
    
    const outputData = new Float32Array(1024);
    await toCPU(ctx, outputTensor, outputData);

    // print values to console
    console.log("WGSL code:", kernelCode.data);
    console.log("Input data:", inputData);
    console.log("Output data:", outputData);
    // update status div showing input and output
    document.getElementById("status").innerText = "WGSL code:\n" + kernelCode.data + "\n\nInput data:\n" + inputData + "\n\nOutput data:\n" + outputData;

    destroyContext(ctx);
}

main().catch(console.error);
