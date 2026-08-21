#include <emscripten.h>

EM_ASYNC_JS(int, runStory, (), {
  try {
    const context = await Module.createContext(false);
    const input = new Int32Array([1, 2, 3, 4]);
    const output = new Int32Array(4);
    const spec = {
      "code": `
        @group(0) @binding(0) var<storage, read> input: array<i32>;
        @group(0) @binding(1) var<storage, read_write> output: array<i32>;
        @compute @workgroup_size({{workgroupSize}})
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          if (id.x < arrayLength(&input)) {
            output[id.x] = input[id.x] * 2 + 1;
          }
        }
      `,
      "workgroupSize": [4, 1, 1],
      "workgroups": [1, 1, 1],
      "bindings": [
        {"data": input, "access": "read"},
        {"data": output, "access": "readWrite"},
      ],
    };

    let rejected = false;
    try {
      await context.run({...spec, "code": "this is not WGSL"});
    } catch (error) {
      rejected = true;
    }
    if (!rejected) throw new Error("invalid WGSL did not reject the run");

    await context.run(spec);
    context.delete();
    if (output.toString() !== "3,5,7,9")
      throw new Error(`unexpected result: ${output}`);
    out("Browser JS binding compute story passed");
    return 0;
  } catch (error) {
    out(`Browser JS binding failed: ${String(error && error.stack || error)}`);
    return 1;
  }
});

int main() { return runStory(); }
