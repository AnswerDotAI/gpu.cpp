#include "gpu.hpp"
#include <array>
#include <cstdio>
#include <future>
#include <memory>

#include "kernels.h"
#include "ops_aot.hpp"
#include "experimental/wgsl.h"      // loopUnrolling

using namespace gpu;

Kernel encoder_forward(Context& ctx, Tensor& out,
                       Tensor& inp, Tensor& wte, Tensor& wpe,
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
  return createKernel(ctx, {kShaderEncoder, 256, kf32},
                      Bindings{inp, wte, wpe, out},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      EncoderParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      });
}

Kernel encoder_backward(Context& ctx, Tensor& dwte, Tensor& dwpe,
                        Tensor& dout, Tensor& inp,
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
  return createKernel(ctx, {kShaderEncoderBackward, 256, kf32},
                      Bindings{dwte, dwpe, dout, inp},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      EncoderParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      });
}

Kernel layernorm_forward(Context& ctx, Tensor& out, Tensor& mean, Tensor& rstd,
                         Tensor& inp, Tensor& weight, Tensor& bias,
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
  return createKernel(ctx, {kShaderLayerNorm, 256, kf32},
                      Bindings{inp, weight, bias, out, mean, rstd},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      LayerNormParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      });
}

Kernel layernorm_backward(Context& ctx, Tensor& dinp, Tensor& dweight, Tensor& dbias,
                          Tensor& dout, Tensor& inp, Tensor& weight, Tensor& mean, Tensor& rstd,
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
  return createKernel(ctx, {kShaderLayerNormBackward, 256, kf32},
                      Bindings{dinp, dweight, dbias, dout, inp, weight, mean, rstd},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      LayerNormParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c)
                      });
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


Kernel matmul_forward(Context& ctx, Tensor& out,
                      const Tensor& inp, const Tensor& weight, const Tensor& bias,
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

  return createKernel(ctx, {kShaderMatmul2D, wgSize, kf32},
                      Bindings{inp, weight, bias, out},
                      nWorkgroups,
                      /* params */
                      MatmulParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(oc)
                      });
}

Kernel matmul_backward(Context& ctx, Tensor& dinp, Tensor& dweight, Tensor& dbias,
                       const Tensor& dout, const Tensor& inp, const Tensor& weight,
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
  return createKernel(ctx, {kShaderMatmulBackward, 256, kf32},
                      Bindings{dinp, dweight, dbias, dout, inp, weight},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      MatmulParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(oc)
                      });
}

Kernel attention_forward(Context& ctx, Tensor& out, Tensor& preatt, Tensor& att,
                         Tensor& inp,
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
  return createKernel(ctx, {kShaderAttention, 256, kf32},
                      Bindings{inp, preatt, att, out},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      AttentionParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(nh)
                      });
}

Kernel attention_backward(Context& ctx, Tensor& dinp, Tensor& dpreatt, Tensor& datt,
                          Tensor& dout, Tensor& inp, Tensor& att,
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
  return createKernel(ctx, {kShaderAttentionBackward, 256, kf32},
                      Bindings{dinp, dpreatt, datt, dout, inp, att},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      AttentionParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(c),
                        static_cast<uint32_t>(nh)
                      });
}

Kernel gelu_forward(Context& ctx, Tensor& out, Tensor& inp, int n) {
  unsigned long N = static_cast<unsigned long>(n);
  setLogLevel(kError);
  return createKernel(ctx, {kShaderGelu, 256, kf32},
                      Bindings{inp, out},
                      /* nWorkgroups */ {cdiv(N, 256), 1, 1});
}

Kernel gelu_backward(Context& ctx, Tensor& dinp, Tensor& inp, Tensor& dout, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  return createKernel(ctx, {kShaderGeluBackward, 256, kf32},
                      Bindings{inp, dout, dinp},
                      /* nWorkgroups */ {cdiv(n, 256), 1, 1});
}

Kernel residual_forward(Context& ctx, Tensor& out, Tensor& inp1, Tensor& inp2, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  return createKernel(ctx, {kShaderResidual, 256, kf32},
                      Bindings{inp1, inp2, out},
                      /* nWorkgroups */ {cdiv(n, 256), 1, 1});
}

Kernel residual_backward(Context& ctx, Tensor& dinp1, Tensor& dinp2, Tensor& dout, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  return createKernel(ctx, {kShaderResidualBackward, 256, kf32},
                      Bindings{dout, dinp1, dinp2},
                      /* nWorkgroups */ {cdiv(n, 256), 1, 1});
}

Kernel softmax_forward(Context& ctx, Tensor& probs, Tensor& logits, int B, int T, int V, int Vp) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
    uint32_t Cp;
  };
  uint32_t b = static_cast<uint32_t>(B);
  uint32_t t = static_cast<uint32_t>(T);
  uint32_t c = static_cast<uint32_t>(V);
  uint32_t cp = static_cast<uint32_t>(Vp);
  assert( (B*T) % 256 == 0);
  return createKernel(
      ctx, {kShaderSoftmax1, 256, kf32}, Bindings{logits, probs},
      Shape{cdiv(B * T, 256), 1, 1}, SoftmaxParam{b * t, c, cp});
}

Kernel crossentropy_forward(Context& ctx, Tensor& losses,
                            Tensor& probs, Tensor& targets,
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
  return createKernel(ctx, {kShaderCrossEntropyForward, 256, kf32},
                      Bindings{losses, probs, targets},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      CrossEntropyParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(vp)
                      });
}

Kernel crossentropy_softmax_backward(Context& ctx, Tensor& dlogits,
                                     Tensor& dlosses, Tensor& probs, Tensor& targets,
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
  return createKernel(ctx, {kShaderCrossEntropySoftmaxBackward, 256, kf32},
                      Bindings{dlogits, dlosses, probs, targets},
                      /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                      /* params */
                      CrossEntropySoftmaxBackwardParams{
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(t),
                        static_cast<uint32_t>(v),
                        static_cast<uint32_t>(vp)
                      });
}
