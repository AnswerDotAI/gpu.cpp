module Main (main) where

import Test.Hspec
import GpuCpp.Types
import GpuCpp
import qualified Data.Vector.Storable as V
import Foreign.C.Types

gelu :: String
gelu= "const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)\n" <>
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

main :: IO ()
main = do
  hspec $ do
    describe "toCPU and toGPU" $ do
      it "writes and reads back" $ do
        context <- createContext
        input <- createTensor context [12] kf32
        toGpu context (V.fromList [1 :: CFloat,2,3,4,1,2,3,4,1,2,3,4]) input
        output <- toCpu context input :: IO (V.Vector CFloat)
        V.toList output `shouldBe` [1,2,3,4,1,2,3,4,1,2,3,4]
    describe "call kernel" $ do
      it "gelu" $ do
        context <- createContext
        input <- createTensor context [12] kf32
        output <- createTensor context [12] kf32
        kernelCode <- createKernelCode gelu 256 kf32
        kernel <- createKernel context kernelCode [input, output] [0,0] [12,1,1]
        toGpu context (V.fromList [1 :: CFloat,2,3,4,1,2,3,4,1,2,3,4]) input
        async <- dispatchKernel context kernel
        wait context async
        vec <- toCpu context output :: IO (V.Vector CFloat)
        V.toList (V.zipWith (\a b -> abs (a - b))
                  vec
                  (V.fromList [0.841192,1.9545977,2.9963627,3.9999297,0.841192,1.9545977,2.9963627,3.9999297,0.841192,1.9545977,2.9963627,3.9999297]))
          `shouldSatisfy` all (< 0.001)
