module Main where

import GpuCpp.Types
import GpuCpp
import qualified Data.Vector.Storable as V
import Foreign.C.Types

main :: IO ()
main = do
  context <- createContext
  input <- createTensor context [12] kf32
  output <- createTensor context [12] kf32
  kernelCode <- createKernelCode
    (
    "const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)\n" <>
    "@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;\n" <>
    "@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;\n" <>
    "@group(0) @binding(1) var<storage, read_write> dummy: array<{{precision}}>;\n" <>
    "@compute @workgroup_size({{workgroupSize}})\n" <>
    "fn main(\n" <>
    "    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {\n" <>
    "    let i: u32 = GlobalInvocationID.x;\n" <>
    "    if (i < arrayLength(&inp)) {\n" <>
    "        let x: f32 = inp[i];\n" <>
    "        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR \n" <>
    "                 * (x + .044715 * x * x * x))), x, x > 10.0);\n" <>
    "    }\n" <>
    "}\n"
    )
    256
    kf32
  kernel <- createKernel context kernelCode [input, output] [0,0] [12,1,1]
  toGpu context (V.fromList [1 :: CFloat,2,3,4,1,2,3,4,1,2,3,4]) input
  async <- dispatchKernel context kernel
  wait context async
  vec <- toCpu context output :: IO (V.Vector CFloat)
  print vec
