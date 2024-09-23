#ifndef OPS_H
#define OPS_H

#include "gpu.hpp"

using namespace gpu;

#ifdef __cplusplus
extern "C" {
#endif

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


void encoder_forward(Context& ctx, float* out,
                     int* inp, float* wte, float* wpe,
                     int B, int T, int C);

void encoder_backward(Context& ctx, float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C);

void layernorm_forward(Context& ctx, float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C);

void layernorm_backward(Context& ctx, float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C);

void matmul_forward(Context& ctx, float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC);

void matmul_backward(Context& ctx, float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC);

void attention_forward(Context& ctx, float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH);

void attention_backward(Context& ctx, float* dinp, float* dpreatt, float* datt,
                         float* dout, float* inp, float* att,
                        int B, int T, int C, int NH);

void gelu_forward(Context& ctx, float* out, float* inp, int N);

void gelu_backward(Context& ctx, float* dinp, float* inp, float* dout, int N);

void residual_forward(Context& ctx, float* out, float* inp1, float* inp2, int N);

void residual_backward(Context& ctx, float* dinp1, float* dinp2, float* dout, int N);

void softmax_forward(Context& ctx, float* probs, float* logits, int B, int T, int V, int Vp);

void crossentropy_forward(Context& ctx, float* losses,
                           float* probs, int* targets,
                           int B, int T, int Vp);

void crossentropy_softmax_backward(Context& ctx, float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp);

#ifdef __cplusplus
}
#endif

#endif // OPS_H
