{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module GpuCpp.Types where

import qualified Language.C.Types as C
import qualified Language.Haskell.TH.Lib as TH
import qualified Data.Map as Map
import Foreign
import qualified Data.Vector.Storable as V

data Context
data Tensor
data Kernel
data KernelCode
data GpuAsync
data StdVector a
 
typeTable :: Map.Map C.TypeSpecifier TH.TypeQ
typeTable = Map.fromList [
        (C.TypeName "gpu::Context", [t|Context|])
      , (C.TypeName "gpu::Tensor", [t|Tensor|])
      , (C.TypeName "gpu::Kernel", [t|Kernel|])
      , (C.TypeName "gpu::KernelCode", [t|KernelCode|])
      , (C.TypeName "GpuAsync", [t|GpuAsync|])
      , (C.TypeName "std::vector", [t|StdVector|])
    ]


class WithVector a b where
  withVector :: [a] -> (Ptr (StdVector b) -> IO c) -> IO c
  
class GpuStorable a where
  toGpu :: ForeignPtr Context -> V.Vector a -> ForeignPtr Tensor -> IO ()
  toCpu :: ForeignPtr Context -> ForeignPtr Tensor -> IO (V.Vector a)

