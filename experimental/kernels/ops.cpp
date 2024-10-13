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
  printf("Creating tensors\n");
  printf("Creating input tensor\%pn", inp);
  Tensor input = createTensor(ctx, Shape{b * t}, ki32, inp);
  printf("Created input tensor\n");
  Tensor wte_t = createTensor(ctx, Shape{v, c}, kf32, wte);
  printf("Created wte tensor\n");
  Tensor wpe_t = createTensor(ctx, Shape{t, c}, kf32, wpe);
  printf("Created wpe tensor\n");
  Tensor output = createTensor(ctx, Shape{b * t * c}, kf32);
  printf("Created tensors\n");
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderEncoder, 256, kf32},
                           Bindings{input, wte_t, wpe_t, output},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           EncoderParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(c)
                           });
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
  Tensor dwte_t = createTensor(ctx, Shape{v, c}, kf32, dwte);
  Tensor dwpe_t = createTensor(ctx, Shape{t, c}, kf32, dwpe);
  Tensor dout_t = createTensor(ctx, Shape{b * t * c}, kf32, dout);
  Tensor input = createTensor(ctx, Shape{b * t}, ki32, inp);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderEncoderBackward, 256, kf32},
                           Bindings{dwte_t, dwpe_t, dout_t, input},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           EncoderParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(c)
                           });
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
  Tensor inp_t = createTensor(ctx, Shape{b * t * c}, kf32, inp);
  Tensor weight_t = createTensor(ctx, Shape{c}, kf32, weight);
  Tensor bias_t = createTensor(ctx, Shape{c}, kf32, bias);
  Tensor out_t = createTensor(ctx, Shape{b * t * c}, kf32);
  Tensor mean_t = createTensor(ctx, Shape{b * t}, kf32);
  Tensor rstd_t = createTensor(ctx, Shape{b * t}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderLayerNorm, 256, kf32},
                           Bindings{inp_t, weight_t, bias_t, out_t, mean_t, rstd_t},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           LayerNormParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(c)
                           });
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
  Tensor dinp_t = createTensor(ctx, Shape{b * t * c}, kf32, dinp);
  Tensor dweight_t = createTensor(ctx, Shape{c}, kf32, dweight);
  Tensor dbias_t = createTensor(ctx, Shape{c}, kf32, dbias);
  Tensor dout_t = createTensor(ctx, Shape{b * t * c}, kf32, dout);
  Tensor inp_t = createTensor(ctx, Shape{b * t * c}, kf32, inp);
  Tensor weight_t = createTensor(ctx, Shape{c}, kf32, weight);
  Tensor mean_t = createTensor(ctx, Shape{b * t}, kf32, mean);
  Tensor rstd_t = createTensor(ctx, Shape{b * t}, kf32, rstd);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderLayerNormBackward, 256, kf32},
                           Bindings{dinp_t, dweight_t, dbias_t, dout_t, inp_t, weight_t, mean_t, rstd_t},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           LayerNormParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(c)
                           });
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_t, dinp, b * t * c * sizeof(float));
  toCPU(ctx, dweight_t, dweight, c * sizeof(float));
  toCPU(ctx, dbias_t, dbias, c * sizeof(float));
}

void matmul_forward(Context& ctx, float* out,
                        const float* inp, const float* weight, const float* bias,
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

  Tensor inp_i = createTensor(ctx, Shape{b * t * c}, kf32, inp);
  Tensor weight_i = createTensor(ctx, Shape{oc * c}, kf32, weight);
  Tensor bias_i = bias == NULL ? createTensor(ctx, Shape{1}, kf32) : createTensor(ctx, Shape{oc}, kf32, bias);
  Tensor out_o = createTensor(ctx, Shape{b * t * oc}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  assert ( (b*t) % 256 == 0 );
  int version = 1;
  if (version == 1){
    static constexpr size_t BT = 64;
    static constexpr size_t BC = 8;
    static constexpr size_t BOC = 64;
    static constexpr size_t TT = BT / BC;
    static constexpr size_t TOC = BOC / BC;
    size_t num_threads = BT * BOC / (TT * TOC);
    Shape wgSize = {num_threads, 1, 1}; // This is the same as BK * BK.
    Shape nWorkgroups = {b, cdiv(T, BT), cdiv(OC, BOC)};
    
    std::string codeString(kShaderMatmul2DTiling);
    replaceAll(codeString, {{"{{precision}}", toString(kf32)},
                            {"{{BT}}", toString(BT)},
                            {"{{BC}}", toString(BC)},
                            {"{{BOC}}", toString(BOC)},
                            {"{{TT}}", toString(TT)},
                            {"{{TOC}}", toString(TOC)},
                            {"{{NUM_TILEI}}", toString(BT * BC / num_threads)},
                            {"{{NUM_TILEW}}", toString(BOC * BC / num_threads)}
                            });
    std::string unrolledCode = loopUnrolling(codeString);
    Kernel op = createKernel(ctx, {unrolledCode, wgSize, kf32},
                             Bindings{inp_i, weight_i, bias_i, out_o},
                             nWorkgroups,
                             /* params */
                             MatmulParams{
                               static_cast<uint32_t>(b),
                               static_cast<uint32_t>(t),
                               static_cast<uint32_t>(c),
                               static_cast<uint32_t>(oc)
                             });
    dispatchKernel(ctx, op, promise);
    wait(ctx, future);
    toCPU(ctx, out_o, out, b * t * oc * sizeof(float));
  } else {
    Kernel op = createKernel(ctx, {kShaderMatmul, 256, kf32},
                             Bindings{inp_i, weight_i, bias_i, out_o},
                             /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                             /* params */
                             MatmulParams{
                               static_cast<uint32_t>(b),
                               static_cast<uint32_t>(t),
                               static_cast<uint32_t>(c),
                               static_cast<uint32_t>(oc)
                             });
    dispatchKernel(ctx, op, promise);
    wait(ctx, future);
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
  Tensor dinp_t = createTensor(ctx, Shape{b * t * c}, kf32, dinp);
  Tensor dweight_t = createTensor(ctx, Shape{oc * c}, kf32, dweight);
  Tensor dbias_t = createTensor(ctx, Shape{oc}, kf32, dbias);
  Tensor dout_t = createTensor(ctx, Shape{b * t * oc}, kf32, dout);
  Tensor inp_t = createTensor(ctx, Shape{b * t * c}, kf32, inp);
  Tensor weight_t = createTensor(ctx, Shape{oc * c}, kf32, weight);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderMatmulBackward, 256, kf32},
                           Bindings{dinp_t, dweight_t, dbias_t, dout_t, inp_t, weight_t},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           MatmulParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(c),
                             static_cast<uint32_t>(oc)
                           });
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
  Tensor inp_t = createTensor(ctx, Shape{b * t * c * 3}, kf32, inp);
  Tensor preatt_t = createTensor(ctx, Shape{b * nh * t * t}, kf32, preatt);
  Tensor att_t = createTensor(ctx, Shape{b * nh * t * t}, kf32, att);
  Tensor out_t = createTensor(ctx, Shape{b * t * c}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderAttention, 256, kf32},
                           Bindings{inp_t, preatt_t, att_t, out_t},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           AttentionParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(c),
                             static_cast<uint32_t>(nh)
                           });
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
  Tensor dinp_t = createTensor(ctx, Shape{b * t * c * 3}, kf32, dinp);
  Tensor dpreatt_t = createTensor(ctx, Shape{b * nh * t * t}, kf32, dpreatt);
  Tensor datt_t = createTensor(ctx, Shape{b * nh * t * t}, kf32, datt);
  Tensor dout_t = createTensor(ctx, Shape{b * t * c}, kf32, dout);
  Tensor inp_t = createTensor(ctx, Shape{b * t * c * 3}, kf32, inp);
  Tensor att_t = createTensor(ctx, Shape{b * nh * t * t}, kf32, att);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderAttentionBackward, 256, kf32},
                           Bindings{dinp_t, dpreatt_t, datt_t, dout_t, inp_t, att_t},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           AttentionParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(c),
                             static_cast<uint32_t>(nh)
                           });
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_t, dinp, b * t * c * 3 * sizeof(float));
  toCPU(ctx, dpreatt_t, dpreatt, b * nh * t * t * sizeof(float));
  toCPU(ctx, datt_t, datt, b * nh * t * t * sizeof(float));
}

void gelu_forward(Context& ctx, float* out, float* inp, int n) {
  unsigned long N = static_cast<unsigned long>(n);
  setLogLevel(kError);
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

void gelu_backward(Context& ctx, float* dinp, float* inp, float* dout, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  Tensor inp_i = createTensor(ctx, Shape{n}, kf32, inp);
  Tensor dout_i = createTensor(ctx, Shape{n}, kf32, dout);
  Tensor dinp_o = createTensor(ctx, Shape{n}, kf32, dinp);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderGeluBackward, 256, kf32},
                           Bindings{inp_i, dout_i, dinp_o},
                           /* nWorkgroups */ {cdiv(n, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dinp_o, dinp, n * sizeof(float));
}

void residual_forward(Context& ctx, float* out, float* inp1, float* inp2, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  Tensor inp1_i = createTensor(ctx, Shape{n}, kf32, inp1);
  Tensor inp2_i = createTensor(ctx, Shape{n}, kf32, inp2);
  Tensor out_o = createTensor(ctx, Shape{n}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderResidual, 256, kf32},
                           Bindings{inp1_i, inp2_i, out_o},
                           /* nWorkgroups */ {cdiv(n, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, out_o, out, n * sizeof(float));
}

void residual_backward(Context& ctx, float* dinp1, float* dinp2, float* dout, int N){
  unsigned long n = static_cast<unsigned long>(N);
  setLogLevel(kError);
  Tensor dout_i = createTensor(ctx, Shape{n}, kf32, dout);
  Tensor dinp1_o = createTensor(ctx, Shape{n}, kf32, dinp1);
  Tensor dinp2_o = createTensor(ctx, Shape{n}, kf32, dinp2);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderResidualBackward, 256, kf32},
                           Bindings{dout_i, dinp1_o, dinp2_o},
                           /* nWorkgroups */ {cdiv(n, 256), 1, 1});
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
  Tensor input = createTensor(ctx, {b * t, cp}, kf32, logits);
  Tensor output = createTensor(ctx, {b * t, cp}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  assert( (B*T) % 256 == 0);
  Kernel op = createKernel(
      ctx, {kShaderSoftmax1, 256, kf32}, Bindings{input, output},
      Shape{cdiv(B * T, 256), 1, 1}, SoftmaxParam{b * t, c, cp});
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
  Tensor losses_t = createTensor(ctx, Shape{b * t}, kf32, losses);
  Tensor probs_t = createTensor(ctx, Shape{b * t * vp}, kf32, probs);
  Tensor targets_t = createTensor(ctx, Shape{b * t}, ki32, targets);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderCrossEntropyForward, 256, kf32},
                           Bindings{losses_t, probs_t, targets_t},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           CrossEntropyParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(vp)
                           });
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
  Tensor dlogits_t = createTensor(ctx, Shape{b * t * vp}, kf32, dlogits);
  Tensor dlosses_t = createTensor(ctx, Shape{b * t}, kf32, dlosses);
  Tensor probs_t = createTensor(ctx, Shape{b * t * vp}, kf32, probs);
  Tensor targets_t = createTensor(ctx, Shape{b * t}, ki32, targets);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderCrossEntropySoftmaxBackward, 256, kf32},
                           Bindings{dlogits_t, dlosses_t, probs_t, targets_t},
                           /* nWorkgroups */ {cdiv(b * t, 256), 1, 1},
                           /* params */
                           CrossEntropySoftmaxBackwardParams{
                             static_cast<uint32_t>(b),
                             static_cast<uint32_t>(t),
                             static_cast<uint32_t>(v),
                             static_cast<uint32_t>(vp)
                           });
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, dlogits_t, dlogits, b * t * vp * sizeof(float));
}
