#ifdef __cplusplus
extern "C" {
#endif

#ifdef METAL_PROFILER
#include "experimental/profiler/metal.hpp"

#define MAIN main_wrapper
static int main_wrapper(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  startCapture();
  int ret = main_wrapper(argc, argv);
  stopCapture();
  return ret;
}

#else
#define MAIN main
#endif

// --  USE_GPU_FOR_* are the GPU/CPU switching flags for the kernels in llm.c. --

#define USE_GPU_FOR_ENCODER_FORWARD 1
// #define USE_GPU_FOR_ENCODER_BACKWARD 1
#define USE_GPU_FOR_LAYERNORM_FORWARD 1
// --  Note: atomicAdd should be used with i32 or u32 not f32.
// #define USE_GPU_FOR_LAYERNORM_BACKWARD 1
// --  Note: matmul_forward kernel works, but it is too slow.
#define USE_GPU_FOR_MATMUL_FORWARD 1
// #define USE_GPU_FOR_MATMUL_BACKWARD 1
#define USE_GPU_FOR_ATTENTION_FORWARD 1
// #define USE_GPU_FOR_ATTENTION_BACKWARD 1
#define USE_GPU_FOR_GELU_FORWARD 1
#define USE_GPU_FOR_GELU_BACKWARD 1
#define USE_GPU_FOR_RESIDUAL_FORWARD 1
#define USE_GPU_FOR_RESIDUAL_BACKWARD 1
#define USE_GPU_FOR_SOFTMAX_FORWARD 1
#define USE_GPU_FOR_CROSSENTROPY_FORWARD 1
#define USE_GPU_FOR_CROSSENTROPY_SOFTMAX_BACKWARD 1


#ifdef USE_GPU_FOR_ENCODER_FORWARD
#define ENCODER_FORWARD_CPU encoder_forward_dummy
#define ENCODER_FORWARD_GPU encoder_forward
#else
#define ENCODER_FORWARD_CPU encoder_forward
#define ENCODER_FORWARD_GPU encoder_forward_dummy
#endif

#ifdef USE_GPU_FOR_ENCODER_BACKWARD
#define ENCODER_BACKWARD_CPU encoder_backward_dummy
#define ENCODER_BACKWARD_GPU encoder_backward
#else
#define ENCODER_BACKWARD_CPU encoder_backward
#define ENCODER_BACKWARD_GPU encoder_backward_dummy
#endif

#ifdef USE_GPU_FOR_LAYERNORM_FORWARD
#define LAYERNORM_FORWARD_CPU layernorm_forward_dummy
#define LAYERNORM_FORWARD_GPU layernorm_forward
#else
#define LAYERNORM_FORWARD_CPU layernorm_forward
#define LAYERNORM_FORWARD_GPU layernorm_forward_dummy
#endif

#ifdef USE_GPU_FOR_LAYERNORM_BACKWARD
#define LAYERNORM_BACKWARD_CPU layernorm_backward_dummy
#define LAYERNORM_BACKWARD_GPU layernorm_backward
#else
#define LAYERNORM_BACKWARD_CPU layernorm_backward
#define LAYERNORM_BACKWARD_GPU layernorm_backward_dummy
#endif

#ifdef USE_GPU_FOR_MATMUL_FORWARD
#define MATMUL_FORWARD_CPU matmul_forward_dummy
#define MATMUL_FORWARD_GPU matmul_forward
#else
#define MATMUL_FORWARD_CPU matmul_forward
#define MATMUL_FORWARD_GPU matmul_forward_dummy
#endif

#ifdef USE_GPU_FOR_MATMUL_BACKWARD
#define MATMUL_BACKWARD_CPU matmul_backward_dummy
#define MATMUL_BACKWARD_GPU matmul_backward
#else
#define MATMUL_BACKWARD_CPU matmul_backward
#define MATMUL_BACKWARD_GPU matmul_backward_dummy
#endif

#ifdef USE_GPU_FOR_ATTENTION_FORWARD
#define ATTENTION_FORWARD_CPU attention_forward_dummy
#define ATTENTION_FORWARD_GPU attention_forward
#else
#define ATTENTION_FORWARD_CPU attention_forward
#define ATTENTION_FORWARD_GPU attention_forward_dummy
#endif

#ifdef USE_GPU_FOR_ATTENTION_BACKWARD
#define ATTENTION_BACKWARD_CPU attention_backward_dummy
#define ATTENTION_BACKWARD_GPU attention_backward
#else
#define ATTENTION_BACKWARD_CPU attention_backward
#define ATTENTION_BACKWARD_GPU attention_backward_dummy
#endif

#ifdef USE_GPU_FOR_GELU_FORWARD
#define GELU_FORWARD_CPU gelu_forward_dummy
#define GELU_FORWARD_GPU gelu_forward
#else
#define GELU_FORWARD_CPU gelu_forward
#define GELU_FORWARD_GPU gelu_forward_dummy
#endif

#ifdef USE_GPU_FOR_GELU_BACKWARD
#define GELU_BACKWARD_CPU gelu_backward_dummy
#define GELU_BACKWARD_GPU gelu_backward
#else
#define GELU_BACKWARD_CPU gelu_backward
#define GELU_BACKWARD_GPU gelu_backward_dummy
#endif

#ifdef USE_GPU_FOR_RESIDUAL_FORWARD
#define RESIDUAL_FORWARD_CPU residual_forward_dummy
#define RESIDUAL_FORWARD_GPU residual_forward
#else
#define RESIDUAL_FORWARD_CPU residual_forward
#define RESIDUAL_FORWARD_GPU residual_forward_dummy
#endif

#ifdef USE_GPU_FOR_RESIDUAL_BACKWARD
#define RESIDUAL_BACKWARD_CPU residual_backward_dummy
#define RESIDUAL_BACKWARD_GPU residual_backward
#else
#define RESIDUAL_BACKWARD_CPU residual_backward
#define RESIDUAL_BACKWARD_GPU residual_backward_dummy
#endif

#ifdef USE_GPU_FOR_SOFTMAX_FORWARD
#define SOFTMAX_FORWARD_CPU softmax_forward_dummy
#define SOFTMAX_FORWARD_GPU softmax_forward
#else
#define SOFTMAX_FORWARD_CPU softmax_forward
#define SOFTMAX_FORWARD_GPU softmax_forward_dummy
#endif

#ifdef USE_GPU_FOR_CROSSENTROPY_FORWARD
#define CROSSENTROPY_FORWARD_CPU crossentropy_forward_dummy
#define CROSSENTROPY_FORWARD_GPU crossentropy_forward
#else
#define CROSSENTROPY_FORWARD_CPU crossentropy_forward
#define CROSSENTROPY_FORWARD_GPU crossentropy_forward_dummy
#endif

#ifdef USE_GPU_FOR_CROSSENTROPY_SOFTMAX_BACKWARD
#define CROSSENTROPY_SOFTMAX_BACKWARD_CPU crossentropy_softmax_backward_dummy
#define CROSSENTROPY_SOFTMAX_BACKWARD_GPU crossentropy_softmax_backward
#else
#define CROSSENTROPY_SOFTMAX_BACKWARD_CPU crossentropy_softmax_backward
#define CROSSENTROPY_SOFTMAX_BACKWARD_GPU crossentropy_softmax_backward_dummy
#endif

void encoder_forward(float* out,
                     int* inp, float* wte, float* wpe,
                     int B, int T, int C);

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C);

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C);

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C);

void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC);

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC);

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH);

void attention_backward(float* dinp, float* dpreatt, float* datt,
                         float* dout, float* inp, float* att,
                        int B, int T, int C, int NH);

void gelu_forward(float* out, float* inp, int N);

void gelu_backward(float* dinp, float* inp, float* dout, int N);

void residual_forward(float* out, float* inp1, float* inp2, int N);

void residual_backward(float* dinp1, float* dinp2, float* dout, int N);

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp);

void crossentropy_forward(float* losses,
                           float* probs, int* targets,
                           int B, int T, int Vp);

void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp);

#ifdef __cplusplus
}
#endif

