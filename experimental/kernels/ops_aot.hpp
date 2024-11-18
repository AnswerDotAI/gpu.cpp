#ifndef OPS_H
#define OPS_H

#include "gpu.hpp"

using namespace gpu;

#define VOCAB_SIZE 50257

// See https://github.com/google/dawn/blob/a8fbe981a86cb59536e2de423d2013a82d9b54a0/src/dawn/native/Limits.cpp
#define LIMITS_BUFFER_SIZE_1GB { \
    .nextInChain = nullptr, \
    .limits = { \
      .maxTextureDimension1D=8192, \
      .maxTextureDimension2D=8192, \
      .maxTextureDimension3D=2048, \
      .maxTextureArrayLayers=256, \
      .maxBindGroups=4, \
      .maxBindGroupsPlusVertexBuffers=24, \
      .maxBindingsPerBindGroup=1000, \
      .maxDynamicUniformBuffersPerPipelineLayout=8, \
      .maxDynamicStorageBuffersPerPipelineLayout=4, \
      .maxSampledTexturesPerShaderStage=16, \
      .maxSamplersPerShaderStage=16, \
      .maxStorageBuffersPerShaderStage=8, \
      .maxStorageTexturesPerShaderStage=4, \
      .maxUniformBuffersPerShaderStage=12, \
      .maxUniformBufferBindingSize=65536, \
      .maxStorageBufferBindingSize=1073741824, \
      .minUniformBufferOffsetAlignment=256, \
      .minStorageBufferOffsetAlignment=256, \
      .maxVertexBuffers=8, \
      .maxBufferSize=0x80000000, \
      .maxVertexAttributes=16, \
      .maxVertexBufferArrayStride=2048, \
      .maxInterStageShaderComponents=64, \
      .maxInterStageShaderVariables=16, \
      .maxColorAttachments=8, \
      .maxColorAttachmentBytesPerSample=32, \
      .maxComputeWorkgroupStorageSize=16384, \
      .maxComputeInvocationsPerWorkgroup=256, \
      .maxComputeWorkgroupSizeX=256, \
      .maxComputeWorkgroupSizeY=256, \
      .maxComputeWorkgroupSizeZ=64, \
      .maxComputeWorkgroupsPerDimension=65535 \
    } \
  }


Kernel encoder_forward(Context& ctx, Tensor& out,
                       Tensor& inp, Tensor& wte, Tensor& wpe,
                       int B, int T, int C);

Kernel encoder_backward(Context& ctx, Tensor& dwte, Tensor& dwpe,
                        Tensor& dout, Tensor& inp,
                        int B, int T, int C);

Kernel layernorm_forward(Context& ctx, Tensor& out, Tensor& mean, Tensor& rstd,
                         Tensor& inp, Tensor& weight, Tensor& bias,
                         int B, int T, int C);

Kernel layernorm_backward(Context& ctx, Tensor& dinp, Tensor& dweight, Tensor& dbias,
                          Tensor& dout, Tensor& inp, Tensor& weight, Tensor& mean, Tensor& rstd,
                          int B, int T, int C);

Kernel matmul_forward(Context& ctx, Tensor& out,
                      const Tensor& inp, const Tensor& weight, const Tensor& bias,
                      int B, int T, int C, int OC);

Kernel matmul_backward(Context& ctx, Tensor& dinp, Tensor& dweight, Tensor& dbias,
                       const Tensor& dout, const Tensor& inp, const Tensor& weight,
                       int B, int T, int C, int OC);

Kernel attention_forward(Context& ctx, Tensor& out, Tensor& preatt, Tensor& att,
                         Tensor& inp,
                         int B, int T, int C, int NH);

Kernel attention_backward(Context& ctx, Tensor& dinp, Tensor& dpreatt, Tensor& datt,
                          Tensor& dout, Tensor& inp, Tensor& att,
                          int B, int T, int C, int NH);

Kernel gelu_forward(Context& ctx, Tensor& out, Tensor& inp, int N);

Kernel gelu_backward(Context& ctx, Tensor& dinp, Tensor& inp, Tensor& dout, int N);

Kernel residual_forward(Context& ctx, Tensor& out, Tensor& inp1, Tensor& inp2, int N);

Kernel residual_backward(Context& ctx, Tensor& dinp1, Tensor& dinp2, Tensor& dout, int N);

Kernel softmax_forward(Context& ctx, Tensor& probs, Tensor& logits, int B, int T, int V, int Vp);

Kernel crossentropy_forward(Context& ctx, Tensor& losses,
                            Tensor& probs, Tensor& targets,
                            int B, int T, int Vp);

Kernel crossentropy_softmax_backward(Context& ctx, Tensor& dlogits,
                                     Tensor& dlosses, Tensor& probs, Tensor& targets,
                                     int B, int T, int V, int Vp);

#endif // OPS_H
