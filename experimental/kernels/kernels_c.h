#ifdef __cplusplus
extern "C" {
#endif

void gelu_forward(float* out, float* inp, int N);
void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp);

#ifdef __cplusplus
}
#endif

