{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module GpuCpp where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import Foreign.C.String
import Foreign.C.Types
import GHC.Int
import GHC.ForeignPtr(mallocPlainForeignPtrBytes)
import Foreign
import Control.Monad (forM_)
import GpuCpp.Types
import Control.Exception.Safe (bracket)
import qualified Data.Vector.Storable as V

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<gpu.hpp>"
C.include "<future>"
C.include "<vector>"

[C.emitBlock|
struct GpuAsync {
  std::promise<void> promise;
  std::future<void> future;
  GpuAsync(): future(promise.get_future()){
  }
};

gpu::Shape vector_to_shape(const std::vector<int64_t> &dims) {
  switch(dims.size()){
  case 1:
    return gpu::Shape{(unsigned long)dims[0]};
    break;
  case 2:
    return gpu::Shape{(unsigned long)dims[0],(unsigned long)dims[1]};
    break;
  case 3:
    return gpu::Shape{(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2]};
    break;
  case 4:
    return gpu::Shape{(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2],(unsigned long)dims[3]};
    break;
  case 5:
    return gpu::Shape{(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2],(unsigned long)dims[3],(unsigned long)dims[4]};
    break;
  }
  return gpu::Shape{0};
}
|]

kf32 :: CInt
kf32 = [C.pure| int { (int)gpu::kf32 } |]

createContext :: IO (ForeignPtr Context)
createContext =
  [C.throwBlock| gpu::Context* { return new gpu::Context(gpu::createContext()); }|] >>=
  newForeignPtr
    [C.funPtr| void deleteContext(gpu::Context* ptr) { delete ptr; }|]


createKernelCode :: String -> CInt -> CInt -> IO (ForeignPtr KernelCode)
createKernelCode kernelString workgroupSize precision =
  withCString kernelString $ \pData ->
    [C.throwBlock| gpu::KernelCode* { return new gpu::KernelCode($(char* pData), $(int workgroupSize), (gpu::NumType)$(int precision)); }|] >>=
    newForeignPtr
      [C.funPtr| void deleteKernelCode(gpu::KernelCode* ptr) { delete ptr; }|]


dispatchKernel :: ForeignPtr Context -> ForeignPtr Kernel -> IO (ForeignPtr GpuAsync)
dispatchKernel context kernel =
  withForeignPtr context $ \c -> 
  withForeignPtr kernel $ \k ->
    [C.throwBlock| GpuAsync* {
      auto async = new GpuAsync();
      gpu::dispatchKernel(*$(gpu::Context* c), *$(gpu::Kernel* k), async->promise);
      return async; }|] >>=
    newForeignPtr
      [C.funPtr| void deleteGpuAsync(GpuAsync* ptr) { delete ptr; }|]
  
wait :: ForeignPtr Context -> ForeignPtr GpuAsync -> IO ()
wait context async =
  withForeignPtr context $ \c -> 
  withForeignPtr async $ \a ->
    [C.throwBlock| void {
      gpu::wait(*$(gpu::Context* c), $(GpuAsync* a)->future);
    }|]

instance WithVector CInt Int64 where
  withVector shape func =
    bracket
      (do
         let len = fromIntegral $ length shape
         vec <- [C.throwBlock| std::vector<int64_t>* {
           return new std::vector<int64_t>($(int len));
         }|]
         ptr <- [C.throwBlock| int64_t* {
           return $(std::vector<int64_t>* vec)->data();
         }|]
         pokeArray ptr (map fromIntegral shape)
         return vec
      ) 
      (\vec -> [C.block| void { delete $(std::vector<int64_t>* vec); }|])
      (\vec -> func vec)

instance WithVector CInt CSize where
  withVector shape func =
    bracket
      (do
         let len = fromIntegral $ length shape
         vec <- [C.throwBlock| std::vector<size_t>* {
           return new std::vector<size_t>($(int len));
         }|]
         ptr <- [C.throwBlock| size_t* {
           return $(std::vector<size_t>* vec)->data();
         }|]
         pokeArray ptr (map fromIntegral shape)
         return vec
      ) 
      (\vec -> [C.block| void { delete $(std::vector<size_t>* vec); }|])
      (\vec -> func vec)

instance WithVector (Ptr Tensor) Tensor where
  withVector ptrs func = 
    bracket (do
                vec <- [C.throwBlock| std::vector<gpu::Tensor>* { return new std::vector<gpu::Tensor>(); }|]
                forM_ ptrs $ do
                  \ptr -> [C.throwBlock| void { $(std::vector<gpu::Tensor>* vec)->push_back(*$(gpu::Tensor* ptr)); }|]
                return vec
            )
            (\vec -> [C.block| void { delete $(std::vector<gpu::Tensor>* vec); }|])
            (\vec -> func vec)

withForeignPtrs :: [ForeignPtr a] -> ([Ptr a] -> IO b) -> IO b
withForeignPtrs [] func = func []
withForeignPtrs (x:xs) func =
  withForeignPtr x $ \x' ->
    withForeignPtrs xs $ \xs' ->
      func (x':xs')

createKernel :: ForeignPtr Context -> ForeignPtr KernelCode -> [ForeignPtr Tensor] -> [Int] -> [Int] -> IO (ForeignPtr Kernel)
createKernel context kernelCode dataBindings viewOffsets totalWorkgroups =
  withForeignPtr context $ \c -> 
  withForeignPtr kernelCode $ \k -> 
  withForeignPtrs dataBindings $ \b ->
  withVector b $ \b' ->
  withVector @CInt (map fromIntegral viewOffsets) $ \v ->
  withVector @CInt (map fromIntegral totalWorkgroups) $ \w ->
    [C.throwBlock| gpu::Kernel* {
      return new gpu::Kernel(gpu::createKernel(
                   *$(gpu::Context* c),
                   *$(gpu::KernelCode* k),
                   $(std::vector<gpu::Tensor>* b')->data(),
                   $(std::vector<gpu::Tensor>* b')->size(),
                   $(std::vector<size_t>* v)->data(),
                   vector_to_shape(*$(std::vector<int64_t>* w))));
    }|] >>=
    newForeignPtr
      [C.funPtr| void deleteKernel(gpu::Kernel* ptr) { delete ptr; }|]
  
createTensor :: ForeignPtr Context -> [CInt] -> CInt -> IO (ForeignPtr Tensor)
createTensor context shape dtype =
  withVector shape $ \s ->
  withForeignPtr context $ \c -> 
    [C.throwBlock| gpu::Tensor* {
      return new gpu::Tensor(gpu::createTensor(*$(gpu::Context* c), vector_to_shape(*$(std::vector<int64_t>* s)), (gpu::NumType)$(int dtype)));
    }|] >>=
    newForeignPtr
      [C.funPtr| void deleteTensor(gpu::Tensor* ptr) { delete ptr; }|]

createVector :: forall a. Storable a => Int -> IO (V.Vector a)
createVector n = do
  ptr <- mallocPlainForeignPtrBytes (n * sizeOf (undefined :: a))
  return $ V.unsafeFromForeignPtr ptr 0 n    
        
instance GpuStorable CFloat where
  toGpu context array tensor =
    withForeignPtr context $ \c -> 
    withForeignPtr tensor $ \t ->
    V.unsafeWith array $ \ptr ->
      [C.throwBlock| void {
        gpu::toGPU(*$(gpu::Context* c), $(float* ptr), *$(gpu::Tensor* t));
      }|]
  toCpu context tensor =
    withForeignPtr context $ \c -> 
    withForeignPtr tensor $ \t -> do
      (size :: CInt) <- [C.block| int {
                                size_t u = sizeof(float);
                                size_t len = $(gpu::Tensor* t)->data.size;
                                return len/u;
                        }|]
      array <- createVector (fromIntegral size)
      V.unsafeWith array $ \ptr ->
        [C.throwBlock| void {
          gpu::toCPU(*$(gpu::Context* c), *$(gpu::Tensor* t), $(float* ptr), $(int size) * sizeof(float));
        }|]
      return array
