#include "gpu.hpp"
#include <array>
#include <cstdio>
#include <future>
#include <memory>

#include "kernels.h"
#include "ops.hpp"
#include "experimental/wgsl.h"      // loopUnrolling

using namespace gpu;

void encoder_forward(Context& ctx, float* out,
                         int* inp, float* wte, float* wpe,
                         int B, int T, int C){
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  unsigned long v = VOCAB_SIZE;
  struct EncoderParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
  };
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "encoder_forward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor input = createTensor(ctx, Shape{b * t}, ki32);
    Tensor wte_t = createTensor(ctx, Shape{v, c}, kf32);
    Tensor wpe_t = createTensor(ctx, Shape{t, c}, kf32);
    Tensor output = createTensor(ctx, Shape{b * t * c}, kf32);
    op = createKernel(ctx, {kShaderEncoder, 256, kf32},
                      Bindings{input, wte_t, wpe_t, output},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      EncoderParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& input = ctx.pool.data[op->buffers[0]];
  Tensor& wte_t = ctx.pool.data[op->buffers[1]];
  Tensor& wpe_t = ctx.pool.data[op->buffers[2]];
  Tensor& output = ctx.pool.data[op->buffers[3]];

  toGPU(ctx, inp, input);
  toGPU(ctx, wte, wte_t);
  toGPU(ctx, wpe, wpe_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, out, b * t * c * sizeof(float));
}

void encoder_backward(Context& ctx, float* dwte, float* dwpe,
                          float* dout, int* inp,
                          int B, int T, int C) {
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  unsigned long v = VOCAB_SIZE;
  struct EncoderParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
  };
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "encoder_backward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor dwte_t = createTensor(ctx, Shape{v, c}, kf32);
    Tensor dwpe_t = createTensor(ctx, Shape{t, c}, kf32);
    Tensor dout_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor input = createTensor(ctx, Shape{b * t}, ki32);
    op = createKernel(ctx, {kShaderEncoderBackward, 256, kf32},
                      Bindings{dwte_t, dwpe_t, dout_t, input},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      EncoderParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& dwte_t = ctx.pool.data[op->buffers[0]];
  Tensor& dwpe_t = ctx.pool.data[op->buffers[1]];
  Tensor& dout_t = ctx.pool.data[op->buffers[2]];
  Tensor& input = ctx.pool.data[op->buffers[3]];

  toGPU(ctx, dwte, dwte_t);
  toGPU(ctx, dwpe, dwpe_t);
  toGPU(ctx, dout, dout_t);
  toGPU(ctx, inp, input);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dwte_t, dwte, v * c * sizeof(float));
  toCPU(ctx, dwpe_t, dwpe, t * c * sizeof(float));
}

void layernorm_forward(Context& ctx, float* out, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C){
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  struct LayerNormParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
  };
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "layernorm_forward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor inp_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor weight_t = createTensor(ctx, Shape{c}, kf32);
    Tensor bias_t = createTensor(ctx, Shape{c}, kf32);
    Tensor out_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor mean_t = createTensor(ctx, Shape{b * t}, kf32);
    Tensor rstd_t = createTensor(ctx, Shape{b * t}, kf32);
    op = createKernel(ctx, {kShaderLayerNorm, 256, kf32},
                      Bindings{inp_t, weight_t, bias_t, out_t, mean_t, rstd_t},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      LayerNormParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& inp_t = ctx.pool.data[op->buffers[0]];
  Tensor& weight_t = ctx.pool.data[op->buffers[1]];
  Tensor& bias_t = ctx.pool.data[op->buffers[2]];
  Tensor& out_t = ctx.pool.data[op->buffers[3]];
  Tensor& mean_t = ctx.pool.data[op->buffers[4]];
  Tensor& rstd_t = ctx.pool.data[op->buffers[5]];

  toGPU(ctx, inp, inp_t);
  toGPU(ctx, weight, weight_t);
  toGPU(ctx, bias, bias_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, out_t, out, b * t * c * sizeof(float));
  toCPU(ctx, mean_t, mean, b * t * sizeof(float));
  toCPU(ctx, rstd_t, rstd, b * t * sizeof(float));
}

void layernorm_backward(Context& ctx, float* dinp, float* dweight, float* dbias,
                            float* dout, float* inp, float* weight, float* mean, float* rstd,
                            int B, int T, int C){
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  struct LayerNormParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
  };
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "layernorm_backward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor dinp_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor dweight_t = createTensor(ctx, Shape{c}, kf32);
    Tensor dbias_t = createTensor(ctx, Shape{c}, kf32);
    Tensor dout_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor inp_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor weight_t = createTensor(ctx, Shape{c}, kf32);
    Tensor mean_t = createTensor(ctx, Shape{b * t}, kf32);
    Tensor rstd_t = createTensor(ctx, Shape{b * t}, kf32);
    op = createKernel(ctx, {kShaderLayerNormBackward, 256, kf32},
                      Bindings{dinp_t, dweight_t, dbias_t, dout_t, inp_t, weight_t, mean_t, rstd_t},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      LayerNormParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& dinp_t = ctx.pool.data[op->buffers[0]];
  Tensor& dweight_t = ctx.pool.data[op->buffers[1]];
  Tensor& dbias_t = ctx.pool.data[op->buffers[2]];
  Tensor& dout_t = ctx.pool.data[op->buffers[3]];
  Tensor& inp_t = ctx.pool.data[op->buffers[4]];
  Tensor& weight_t = ctx.pool.data[op->buffers[5]];
  Tensor& mean_t = ctx.pool.data[op->buffers[6]];
  Tensor& rstd_t = ctx.pool.data[op->buffers[7]];

  toGPU(ctx, dinp, dinp_t);
  toGPU(ctx, dweight, dweight_t);
  toGPU(ctx, dbias, dbias_t);
  toGPU(ctx, dout, dout_t);
  toGPU(ctx, inp, inp_t);
  toGPU(ctx, weight, weight_t);
  toGPU(ctx, mean, mean_t);
  toGPU(ctx, rstd, rstd_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_t, dinp, b * t * c * sizeof(float));
  toCPU(ctx, dweight_t, dweight, c * sizeof(float));
  toCPU(ctx, dbias_t, dbias, c * sizeof(float));
}

struct DurationTime {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::chrono::microseconds duration;
  std::string src;
  bool verbose;
  
  inline DurationTime(const std::string& src, bool verbose = true) {
    this->src = src;
    this->verbose = verbose;
    start = std::chrono::high_resolution_clock::now();
  }

  inline ~DurationTime() {
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (this->verbose) {
      printf("Duration(%s): %.1f microseconds\n", src.c_str(), static_cast<double>(duration.count()));
    }
  }
};


void matmul_forward(Context& ctx, float* out,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC){
  bool verbose = false;
  DurationTime duration("matmul_forward_gpu", verbose);
  struct MatmulParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t OC;
  };
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  unsigned long oc = static_cast<unsigned long>(OC);
  setLogLevel(kError);

  // Generate the key of the cache by arguments.
  std::string key = "matmul_forward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C) + "_" + std::to_string(OC);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    constexpr size_t BT = 64;
    constexpr size_t BC = 8;
    constexpr size_t BOC = 64;
    constexpr size_t TT = BT / BC;
    constexpr size_t TOC = BOC / BC;
    size_t num_threads = BT * BOC / (TT * TOC);
    Shape wgSize = {num_threads, 1, 1};
    Shape nWorkgroups = {b, cdiv(T, BT), cdiv(OC, BOC)};

    std::string kShaderMatmul2DTiling_(kShaderMatmul2DTiling);
    std::string kShaderMatmul2D(loopUnrolling(
                                              replaceAll(kShaderMatmul2DTiling_,
                                                         {{"{{precision}}", toString(kf32)},
                                                          {"{{BT}}", toString(BT)},
                                                          {"{{BC}}", toString(BC)},
                                                          {"{{BOC}}", toString(BOC)},
                                                          {"{{TT}}", toString(TT)},
                                                          {"{{TOC}}", toString(TOC)},
                                                          {"{{NUM_TILEI}}", toString(BT * BC / num_threads)},
                                                          {"{{NUM_TILEW}}", toString(BOC * BC / num_threads)}
                                                         })
                                              )
                                );

    Tensor inp_i = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor weight_i = createTensor(ctx, Shape{oc * c}, kf32);
    Tensor bias_i = bias == NULL ? createTensor(ctx, Shape{1}, kf32) : createTensor(ctx, Shape{oc}, kf32);
    Tensor out_o = createTensor(ctx, Shape{b * t * oc}, kf32);
    
    op = createKernel(ctx, {kShaderMatmul2D, wgSize, kf32},
                      Bindings{inp_i, weight_i, bias_i, out_o},
                      nWorkgroups,
                      /* params */
                      MatmulParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(oc)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& inp_i = ctx.pool.data[op->buffers[0]];
  Tensor& weight_i = ctx.pool.data[op->buffers[1]];
  Tensor& bias_i = ctx.pool.data[op->buffers[2]];
  Tensor& out_o = ctx.pool.data[op->buffers[3]];

  toGPU(ctx, inp, inp_i);
  toGPU(ctx, weight, weight_i);
  if (bias != NULL) {
    toGPU(ctx, bias, bias_i);
  }

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  {
    DurationTime duration("matmul_forward_gpu without creating tensors", verbose);
    {
      DurationTime duration("matmul_forward_gpu without creating kernel", verbose);
      dispatchKernel(ctx, op, promise);
      wait(ctx, future);
      toCPU(ctx, out_o, out, b * t * oc * sizeof(float));
    }
  }
  toCPU(ctx, out_o, out, b * t * oc * sizeof(float));
}

void matmul_backward(Context& ctx, float* dinp, float* dweight, float* dbias,
                         const float* dout, const float* inp, const float* weight,
                         int B, int T, int C, int OC){
  struct MatmulParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t OC;
  };
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  unsigned long oc = static_cast<unsigned long>(OC);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "matmul_backward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C) + "_" + std::to_string(OC);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor dinp_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor dweight_t = createTensor(ctx, Shape{oc * c}, kf32);
    Tensor dbias_t = createTensor(ctx, Shape{oc}, kf32);
    Tensor dout_t = createTensor(ctx, Shape{b * t * oc}, kf32);
    Tensor inp_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor weight_t = createTensor(ctx, Shape{oc * c}, kf32);
    op = createKernel(ctx, {kShaderMatmulBackward, 256, kf32},
                      Bindings{dinp_t, dweight_t, dbias_t, dout_t, inp_t, weight_t},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      MatmulParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(oc)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& dinp_t = ctx.pool.data[op->buffers[0]];
  Tensor& dweight_t = ctx.pool.data[op->buffers[1]];
  Tensor& dbias_t = ctx.pool.data[op->buffers[2]];
  Tensor& dout_t = ctx.pool.data[op->buffers[3]];
  Tensor& inp_t = ctx.pool.data[op->buffers[4]];
  Tensor& weight_t = ctx.pool.data[op->buffers[5]];

  toGPU(ctx, dinp, dinp_t);
  toGPU(ctx, dweight, dweight_t);
  toGPU(ctx, dbias, dbias_t);
  toGPU(ctx, dout, dout_t);
  toGPU(ctx, inp, inp_t);
  toGPU(ctx, weight, weight_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_t, dinp, b * t * c * sizeof(float));
  toCPU(ctx, dweight_t, dweight, oc * c * sizeof(float));
  toCPU(ctx, dbias_t, dbias, oc * sizeof(float));
}

void attention_forward(Context& ctx, float* out, float* preatt, float* att,
                           float* inp,
                           int B, int T, int C, int NH){
  struct AttentionParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t NH;
  };
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  unsigned long nh = static_cast<unsigned long>(NH);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "attention_forward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C) + "_" + std::to_string(NH);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor inp_t = createTensor(ctx, Shape{b * t * c * 3}, kf32);
    Tensor preatt_t = createTensor(ctx, Shape{b * nh * t * t}, kf32);
    Tensor att_t = createTensor(ctx, Shape{b * nh * t * t}, kf32);
    Tensor out_t = createTensor(ctx, Shape{b * t * c}, kf32);
    op = createKernel(ctx, {kShaderAttention, 256, kf32},
                      Bindings{inp_t, preatt_t, att_t, out_t},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      AttentionParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(nh)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& inp_t = ctx.pool.data[op->buffers[0]];
  Tensor& preatt_t = ctx.pool.data[op->buffers[1]];
  Tensor& att_t = ctx.pool.data[op->buffers[2]];
  Tensor& out_t = ctx.pool.data[op->buffers[3]];

  toGPU(ctx, inp, inp_t);
  toGPU(ctx, preatt, preatt_t);
  toGPU(ctx, att, att_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, preatt_t, preatt, b * nh * t * t * sizeof(float));
  toCPU(ctx, att_t, att, b * nh * t * t * sizeof(float));
  toCPU(ctx, out_t, out, b * t * c * sizeof(float));
}

void attention_backward(Context& ctx, float* dinp, float* dpreatt, float* datt,
                            float* dout, float* inp, float* att,
                            int B, int T, int C, int NH){
  struct AttentionParams {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t NH;
  };
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long c = static_cast<unsigned long>(C);
  unsigned long nh = static_cast<unsigned long>(NH);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "attention_backward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(C) + "_" + std::to_string(NH);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor dinp_t = createTensor(ctx, Shape{b * t * c * 3}, kf32);
    Tensor dpreatt_t = createTensor(ctx, Shape{b * nh * t * t}, kf32);
    Tensor datt_t = createTensor(ctx, Shape{b * nh * t * t}, kf32);
    Tensor dout_t = createTensor(ctx, Shape{b * t * c}, kf32);
    Tensor inp_t = createTensor(ctx, Shape{b * t * c * 3}, kf32);
    Tensor att_t = createTensor(ctx, Shape{b * nh * t * t}, kf32);
    op = createKernel(ctx, {kShaderAttentionBackward, 256, kf32},
                      Bindings{dinp_t, dpreatt_t, datt_t, dout_t, inp_t, att_t},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      AttentionParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(nh)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& dinp_t = ctx.pool.data[op->buffers[0]];
  Tensor& dpreatt_t = ctx.pool.data[op->buffers[1]];
  Tensor& datt_t = ctx.pool.data[op->buffers[2]];
  Tensor& dout_t = ctx.pool.data[op->buffers[3]];
  Tensor& inp_t = ctx.pool.data[op->buffers[4]];
  Tensor& att_t = ctx.pool.data[op->buffers[5]];

  toGPU(ctx, dinp, dinp_t);
  toGPU(ctx, dpreatt, dpreatt_t);
  toGPU(ctx, datt, datt_t);
  toGPU(ctx, dout, dout_t);
  toGPU(ctx, inp, inp_t);
  toGPU(ctx, att, att_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_t, dinp, b * t * c * 3 * sizeof(float));
  toCPU(ctx, dpreatt_t, dpreatt, b * nh * t * t * sizeof(float));
  toCPU(ctx, datt_t, datt, b * nh * t * t * sizeof(float));
}

void gelu_forward(Context& ctx, float* out, float* inp, int n) {
  unsigned long N = static_cast<unsigned long>(n);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "gelu_forward_" + std::to_string(n);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor input = createTensor(ctx, Shape{N}, kf32);
    Tensor output = createTensor(ctx, Shape{N}, kf32);
    op = createKernel(ctx, {kShaderGelu, 256, kf32},
                      Bindings{input, output},
                      /* nWorkgroups */ {cdiv(N, 256), 1, 1},
                      nullptr,
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& input = ctx.pool.data[op->buffers[0]];
  Tensor& output = ctx.pool.data[op->buffers[1]];

  toGPU(ctx, inp, input);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, out, N * sizeof(float));
}

void gelu_backward(Context& ctx, float* dinp, float* inp, float* dout, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "gelu_backward_" + std::to_string(N);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor inp_i = createTensor(ctx, Shape{n}, kf32);
    Tensor dout_i = createTensor(ctx, Shape{n}, kf32);
    Tensor dinp_o = createTensor(ctx, Shape{n}, kf32);
    op = createKernel(ctx, {kShaderGeluBackward, 256, kf32},
                      Bindings{inp_i, dout_i, dinp_o},
                      /* nWorkgroups */ {cdiv(n, 256), 1, 1},
                      nullptr,
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& inp_i = ctx.pool.data[op->buffers[0]];
  Tensor& dout_i = ctx.pool.data[op->buffers[1]];
  Tensor& dinp_o = ctx.pool.data[op->buffers[2]];

  toGPU(ctx, inp, inp_i);
  toGPU(ctx, dout, dout_i);
  toGPU(ctx, dinp, dinp_o);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_o, dinp, n * sizeof(float));
}

void residual_forward(Context& ctx, float* out, float* inp1, float* inp2, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "residual_forward_" + std::to_string(N);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor inp1_i = createTensor(ctx, Shape{n}, kf32);
    Tensor inp2_i = createTensor(ctx, Shape{n}, kf32);
    Tensor out_o = createTensor(ctx, Shape{n}, kf32);
    op = createKernel(ctx, {kShaderResidual, 256, kf32},
                      Bindings{inp1_i, inp2_i, out_o},
                      /* nWorkgroups */ {cdiv(n, 256), 1, 1},
                      nullptr,
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& inp1_i = ctx.pool.data[op->buffers[0]];
  Tensor& inp2_i = ctx.pool.data[op->buffers[1]];
  Tensor& out_o = ctx.pool.data[op->buffers[2]];

  toGPU(ctx, inp1, inp1_i);
  toGPU(ctx, inp2, inp2_i);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, out_o, out, n * sizeof(float));
}

void residual_backward(Context& ctx, float* dinp1, float* dinp2, float* dout, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "residual_backward_" + std::to_string(N);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor dout_i = createTensor(ctx, Shape{n}, kf32);
    Tensor dinp1_o = createTensor(ctx, Shape{n}, kf32);
    Tensor dinp2_o = createTensor(ctx, Shape{n}, kf32);
    op = createKernel(ctx, {kShaderResidualBackward, 256, kf32},
                      Bindings{dout_i, dinp1_o, dinp2_o},
                      /* nWorkgroups */ {cdiv(n, 256), 1, 1},
                      nullptr,
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& dout_i = ctx.pool.data[op->buffers[0]];
  Tensor& dinp1_o = ctx.pool.data[op->buffers[1]];
  Tensor& dinp2_o = ctx.pool.data[op->buffers[2]];

  toGPU(ctx, dout, dout_i);
  toGPU(ctx, dinp1, dinp1_o);
  toGPU(ctx, dinp2, dinp2_o);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp1_o, dinp1, n * sizeof(float));
  toCPU(ctx, dinp2_o, dinp2, n * sizeof(float));
}

void softmax_forward(Context& ctx, float* probs, float* logits, int B, int T, int V, int Vp) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
    uint32_t Cp;
  };
  uint32_t b = static_cast<uint32_t>(B);
  uint32_t t = static_cast<uint32_t>(T);
  uint32_t c = static_cast<uint32_t>(V);
  uint32_t cp = static_cast<uint32_t>(Vp);
  // Generate the key of the cache by arguments.
  std::string key = "softmax_forward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(V) + "_" + std::to_string(Vp);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor input = createTensor(ctx, {b * t, cp}, kf32);
    Tensor output = createTensor(ctx, {b * t, cp}, kf32);
    assert( (B*T) % 256 == 0);
    op = createKernel(
        ctx, {kShaderSoftmax1, 256, kf32}, Bindings{input, output},
        Shape{cdiv(B * T, 256), 1, 1}, SoftmaxParam{b * t, c, cp},
        nullptr,
        key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& input = ctx.pool.data[op->buffers[0]];
  Tensor& output = ctx.pool.data[op->buffers[1]];

  toGPU(ctx, logits, input);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, probs, sizeof(float)*b*t*cp);
}

void crossentropy_forward(Context& ctx, float* losses,
                              float* probs, int* targets,
                              int B, int T, int Vp){
  struct CrossEntropyParams {
    uint32_t B;
    uint32_t T;
    uint32_t VP;
  };
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long vp = static_cast<unsigned long>(Vp);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "crossentropy_forward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(Vp);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor losses_t = createTensor(ctx, Shape{b * t}, kf32);
    Tensor probs_t = createTensor(ctx, Shape{b * t * vp}, kf32);
    Tensor targets_t = createTensor(ctx, Shape{b * t}, ki32);
    op = createKernel(ctx, {kShaderCrossEntropyForward, 256, kf32},
                      Bindings{losses_t, probs_t, targets_t},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      CrossEntropyParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(vp)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& losses_t = ctx.pool.data[op->buffers[0]];
  Tensor& probs_t = ctx.pool.data[op->buffers[1]];
  Tensor& targets_t = ctx.pool.data[op->buffers[2]];

  toGPU(ctx, losses, losses_t);
  toGPU(ctx, probs, probs_t);
  toGPU(ctx, targets, targets_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, losses_t, losses, b * t * sizeof(float));
}

void crossentropy_softmax_backward(Context& ctx, float* dlogits,
                                       float* dlosses, float* probs, int* targets,
                                       int B, int T, int V, int Vp){
  struct CrossEntropySoftmaxBackwardParams {
    uint32_t B;
    uint32_t T;
    uint32_t V;
    uint32_t VP;
  };
  unsigned long b = static_cast<unsigned long>(B);
  unsigned long t = static_cast<unsigned long>(T);
  unsigned long v = static_cast<unsigned long>(V);
  unsigned long vp = static_cast<unsigned long>(Vp);
  setLogLevel(kError);
  // Generate the key of the cache by arguments.
  std::string key = "crossentropy_softmax_backward_" + std::to_string(B) + "_" + std::to_string(T) + "_" + std::to_string(V) + "_" + std::to_string(Vp);
  Kernel op;
  if (ctx.kernelPool.data.find(key) == ctx.kernelPool.data.end()) {
    Tensor dlogits_t = createTensor(ctx, Shape{b * t * vp}, kf32);
    Tensor dlosses_t = createTensor(ctx, Shape{b * t}, kf32);
    Tensor probs_t = createTensor(ctx, Shape{b * t * vp}, kf32);
    Tensor targets_t = createTensor(ctx, Shape{b * t}, ki32);
    op = createKernel(ctx, {kShaderCrossEntropySoftmaxBackward, 256, kf32},
                      Bindings{dlogits_t, dlosses_t, probs_t, targets_t},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      CrossEntropySoftmaxBackwardParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(v),
                        static_cast<uint32_t>(vp)
                      },
                      nullptr,
                      key.c_str());
  } else {
    op = ctx.kernelPool.data[key];
  }
  Tensor& dlogits_t = ctx.pool.data[op->buffers[0]];
  Tensor& dlosses_t = ctx.pool.data[op->buffers[1]];
  Tensor& probs_t = ctx.pool.data[op->buffers[2]];
  Tensor& targets_t = ctx.pool.data[op->buffers[3]];

  toGPU(ctx, dlogits, dlogits_t);
  toGPU(ctx, dlosses, dlosses_t);
  toGPU(ctx, probs, probs_t);
  toGPU(ctx, targets, targets_t);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dlogits_t, dlogits, b * t * vp * sizeof(float));
}
