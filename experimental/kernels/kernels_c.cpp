#include "gpu.h"
#include <array>
#include <cstdio>
#include <future>

#include "kernels.h"
#include "kernels_c.h"

using namespace gpu; // createContext, createTensor, createKernel,
                     // createShader, dispatchKernel, wait, toCPU
                     // Tensor, Kernel, Context, Shape, kf32



void ENCODER_FORWARD_GPU(float* out,
                         int* inp, float* wte, float* wpe,
                         int B, int T, int C){
}

void ENCODER_BACKWARD_GPU(float* dwte, float* dwpe,
                          float* dout, int* inp,
                          int B, int T, int C){
}

void LAYERNORM_FORWARD_GPU(float* out, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C){
}

void LAYERNORM_BACKWARD_GPU(float* dinp, float* dweight, float* dbias,
                            float* dout, float* inp, float* weight, float* mean, float* rstd,
                            int B, int T, int C){
}

void MATMUL_FORWARD_GPU(float* out,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC){
}

void MATMUL_BACKWARD_GPU(float* dinp, float* dweight, float* dbias,
                         const float* dout, const float* inp, const float* weight,
                         int B, int T, int C, int OC){
}

void ATTENTION_FORWARD_GPU(float* out, float* preatt, float* att,
                           float* inp,
                           int B, int T, int C, int NH){
}

void ATTENTION_BACKWARD_GPU(float* dinp, float* dpreatt, float* datt,
                            float* dout, float* inp, float* att,
                            int B, int T, int C, int NH){
}

void GELU_FORWARD_GPU(float* out, float* inp, int n) {
  unsigned long N = static_cast<unsigned long>(n);
  setLogLevel(kError);
  Context ctx = createContext();
  Tensor input = createTensor(ctx, Shape{N}, kf32, inp);
  Tensor output = createTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderGelu, 256, kf32},
                           Bindings{input, output},
                           /* nWorkgroups */ {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, out, N * sizeof(float));
}

void GELU_BACKWARD_GPU(float* dinp, float* inp, float* dout, int n){
  unsigned long N = static_cast<unsigned long>(n);
  setLogLevel(kError);
  Context ctx = createContext();
  Tensor inp_i = createTensor(ctx, Shape{N}, kf32, inp);
  Tensor dout_i = createTensor(ctx, Shape{N}, kf32, dout);
  Tensor dinp_o = createTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderGeluBackward, 256, kf32},
                           Bindings{inp_i, dout_i, dinp_o},
                           /* nWorkgroups */ {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_o, dinp, N * sizeof(float));
}

void RESIDUAL_FORWARD_GPU(float* out, float* inp1, float* inp2, int N){
}

void RESIDUAL_BACKWARD_GPU(float* dinp1, float* dinp2, float* dout, int N){
}

void SOFTMAX_FORWARD_GPU(float* probs, float* logits, int b, int t, int v, int vp) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  uint32_t B = static_cast<uint32_t>(b);
  uint32_t T = static_cast<uint32_t>(t);
  uint32_t C = static_cast<uint32_t>(v);
  Context ctx = createContext();
  Tensor input = createTensor(ctx, {B * T, C}, kf32, logits);
  Tensor output = createTensor(ctx, {B * T, C}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(
      ctx, {kShaderSoftmax1, 256, kf32}, Bindings{input, output},
      Shape{cdiv(B * T, 256), 1, 1}, SoftmaxParam{B * T, C});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, probs, sizeof(float)*B*T*C);
}

void CROSSENTROPY_FORWARD_GPU(float* losses,
                              float* probs, int* targets,
                              int B, int T, int Vp){
}

void CROSSENTROPY_SOFTMAX_BACKWARD_GPU(float* dlogits,
                                       float* dlosses, float* probs, int* targets,
                                       int B, int T, int V, int Vp){
}

