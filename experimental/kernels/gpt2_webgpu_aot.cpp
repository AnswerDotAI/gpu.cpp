#include "gpu.hpp"
#include "ops_aot.hpp"
/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <memory>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

using namespace gpu;

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    Tensor wte; // (V, C)
    Tensor wpe; // (maxT, C)
    std::vector<Tensor> ln1w; // (L, C)
    std::vector<Tensor> ln1b; // (L, C)
    std::vector<Tensor> qkvw; // (L, 3*C, C)
    std::vector<Tensor> qkvb; // (L, 3*C)
    std::vector<Tensor> attprojw; // (L, C, C)
    std::vector<Tensor> attprojb; // (L, C)
    std::vector<Tensor> ln2w; // (L, C)
    std::vector<Tensor> ln2b; // (L, C)
    std::vector<Tensor> fcw; // (L, 4*C, C)
    std::vector<Tensor> fcb; // (L, 4*C)
    std::vector<Tensor> fcprojw; // (L, C, 4*C)
    std::vector<Tensor> fcprojb; // (L, C)
    Tensor lnfw; // (C)
    Tensor lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
void malloc_and_point_parameters(Context& ctx, GPT2Config config, ParameterTensors* params, size_t* param_sizes) {
    size_t L = config.num_layers;
    params->wte = createTensor(ctx, Shape{param_sizes[0]}, kf32);
    params->wpe = createTensor(ctx, Shape{param_sizes[1]}, kf32);

    params->ln1w.resize(L);
    params->ln1b.resize(L);
    params->qkvw.resize(L);
    params->qkvb.resize(L);
    params->attprojw.resize(L);
    params->attprojb.resize(L);
    params->ln2w.resize(L);
    params->ln2b.resize(L);
    params->fcw.resize(L);
    params->fcb.resize(L);
    params->fcprojw.resize(L);
    params->fcprojb.resize(L);
    for(int l = 0; l < L ; l++) {
      params->ln1w[l] = createTensor(ctx, Shape{param_sizes[2]/config.num_layers}, kf32);
      params->ln1b[l] = createTensor(ctx, Shape{param_sizes[3]/config.num_layers}, kf32);
      params->qkvw[l] = createTensor(ctx, Shape{param_sizes[4]/config.num_layers}, kf32);
      params->qkvb[l] = createTensor(ctx, Shape{param_sizes[5]/config.num_layers}, kf32);
      params->attprojw[l] = createTensor(ctx, Shape{param_sizes[6]/config.num_layers}, kf32);
      params->attprojb[l] = createTensor(ctx, Shape{param_sizes[7]/config.num_layers}, kf32);
      params->ln2w[l] = createTensor(ctx, Shape{param_sizes[8]/config.num_layers}, kf32);
      params->ln2b[l] = createTensor(ctx, Shape{param_sizes[9]/config.num_layers}, kf32);
      params->fcw[l] = createTensor(ctx, Shape{param_sizes[10]/config.num_layers}, kf32);
      params->fcb[l] = createTensor(ctx, Shape{param_sizes[11]/config.num_layers}, kf32);
      params->fcprojw[l] = createTensor(ctx, Shape{param_sizes[12]/config.num_layers}, kf32);
      params->fcprojb[l] = createTensor(ctx, Shape{param_sizes[13]/config.num_layers}, kf32);
    }
    params->lnfw = createTensor(ctx, Shape{param_sizes[14]}, kf32);
    params->lnfb = createTensor(ctx, Shape{param_sizes[15]}, kf32);
}


#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    Tensor encoded; // (B, T, C)
    std::vector<Tensor> ln1; // (L, B, T, C)
    std::vector<Tensor> ln1_mean; // (L, B, T)
    std::vector<Tensor> ln1_rstd; // (L, B, T)
    std::vector<Tensor> qkv; // (L, B, T, 3*C)
    std::vector<Tensor> atty; // (L, B, T, C)
    std::vector<Tensor> preatt; // (L, B, NH, T, T)
    std::vector<Tensor> att; // (L, B, NH, T, T)
    std::vector<Tensor> attproj; // (L, B, T, C)
    std::vector<Tensor> residual2; // (L, B, T, C)
    std::vector<Tensor> ln2; // (L, B, T, C)
    std::vector<Tensor> ln2_mean; // (L, B, T)
    std::vector<Tensor> ln2_rstd; // (L, B, T)
    std::vector<Tensor> fch; // (L, B, T, 4*C)
    std::vector<Tensor> fch_gelu; // (L, B, T, 4*C)
    std::vector<Tensor> fcproj; // (L, B, T, C)
    std::vector<Tensor> residual3; // (L, B, T, C)
    Tensor lnf; // (B, T, C)
    Tensor lnf_mean; // (B, T)
    Tensor lnf_rstd; // (B, T)
    Tensor logits; // (B, T, V)
    Tensor probs; // (B, T, V)
    Tensor losses; // (B, T)
} ActivationTensors;

typedef struct {
    Kernel encoder_forward;
    std::vector<Kernel> layernorm_forward;
    std::vector<Kernel> qkv_projection_forward;
    std::vector<Kernel> attention_forward;
    std::vector<Kernel> attention_projection_forward;
    std::vector<Kernel> residual_forward;
    std::vector<Kernel> ff_up_forward;
    std::vector<Kernel> gelu_forward;
    std::vector<Kernel> ff_down_forward;
    std::vector<Kernel> residual2_forward;
    Kernel layernorm_final_forward;
    Kernel matmul_final_forward;
    Kernel softmax_final_forward;
    Kernel crossentropy_forward;
  
    Kernel crossentropy_softmax_backward;
    Kernel matmul_final_backward;
    Kernel layernorm_final_backward;
    std::vector<Kernel> residual2_backward;
    std::vector<Kernel> ff_down_backward;
    std::vector<Kernel> gelu_backward;
    std::vector<Kernel> ff_up_backward;
    std::vector<Kernel> layernorm2_backward;
    std::vector<Kernel> attention_projection_backward;
    std::vector<Kernel> attention_backward;
    std::vector<Kernel> qkv_projection_backward;
    std::vector<Kernel> layernorm1_backward;
    Kernel encoder_backward;
} Kernels;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

void malloc_and_point_activations(Context& ctx, GPT2Config config, ActivationTensors* acts, size_t* act_sizes) {
    size_t L = config.num_layers;
    acts->encoded = createTensor(ctx, Shape{act_sizes[0]}, kf32);
    acts->ln1.resize(L);
    acts->ln1_mean.resize(L);
    acts->ln1_rstd.resize(L);
    acts->qkv.resize(L);
    acts->atty.resize(L);
    acts->preatt.resize(L);
    acts->att.resize(L);
    acts->attproj.resize(L);
    acts->residual2.resize(L);
    acts->ln2.resize(L);
    acts->ln2_mean.resize(L);
    acts->ln2_rstd.resize(L);
    acts->fch.resize(L);
    acts->fch_gelu.resize(L);
    acts->fcproj.resize(L);
    acts->residual3.resize(L);
    for (int l = 0; l < L; l++) {
        acts->ln1[l] = createTensor(ctx, Shape{act_sizes[1]/config.num_layers}, kf32);
        acts->ln1_mean[l] = createTensor(ctx, Shape{act_sizes[2]/config.num_layers}, kf32);
        acts->ln1_rstd[l] = createTensor(ctx, Shape{act_sizes[3]/config.num_layers}, kf32);
        acts->qkv[l] = createTensor(ctx, Shape{act_sizes[4]/config.num_layers}, kf32);
        acts->atty[l] = createTensor(ctx, Shape{act_sizes[5]/config.num_layers}, kf32);
        acts->preatt[l] = createTensor(ctx, Shape{act_sizes[6]/config.num_layers}, kf32);
        acts->att[l] = createTensor(ctx, Shape{act_sizes[7]/config.num_layers}, kf32);
        acts->attproj[l] = createTensor(ctx, Shape{act_sizes[8]/config.num_layers}, kf32);
        acts->residual2[l] = createTensor(ctx, Shape{act_sizes[9]/config.num_layers}, kf32);
        acts->ln2[l] = createTensor(ctx, Shape{act_sizes[10]/config.num_layers}, kf32);
        acts->ln2_mean[l] = createTensor(ctx, Shape{act_sizes[11]/config.num_layers}, kf32);
        acts->ln2_rstd[l] = createTensor(ctx, Shape{act_sizes[12]/config.num_layers}, kf32);
        acts->fch[l] = createTensor(ctx, Shape{act_sizes[13]/config.num_layers}, kf32);
        acts->fch_gelu[l] = createTensor(ctx, Shape{act_sizes[14]/config.num_layers}, kf32);
        acts->fcproj[l] = createTensor(ctx, Shape{act_sizes[15]/config.num_layers}, kf32);
        acts->residual3[l] = createTensor(ctx, Shape{act_sizes[16]/config.num_layers}, kf32);
    }
    acts->lnf = createTensor(ctx, Shape{act_sizes[17]}, kf32);
    acts->lnf_mean = createTensor(ctx, Shape{act_sizes[18]}, kf32);
    acts->lnf_rstd = createTensor(ctx, Shape{act_sizes[19]}, kf32);
    acts->logits = createTensor(ctx, Shape{act_sizes[20]}, kf32);
    acts->probs = createTensor(ctx, Shape{act_sizes[21]}, kf32);
    acts->losses = createTensor(ctx, Shape{act_sizes[22]}, kf32);
}

void gpu_alloc(Context& ctx, Tensor* tensors, size_t* sizes, size_t n) { 
    for (size_t i = 0; i < n; i++) {
        tensors[i] = createTensor(ctx, Shape{sizes[i]}, kf32);
    }
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    Tensor inputs; // the input tokens for the current forward pass
    Tensor targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    float* mean_loss_buffer;
    float* probs_buffer;

    Tensor nullTensor;

    // kernels
    Kernels kernels;
    bool backward_enabled;
} GPT2;

void gpt2_build_from_checkpoint(Context& ctx, GPT2 *model, const char* checkpoint_path) {
    printf("Building GPT-2 model from checkpoint '%s'\n", checkpoint_path);
    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
#ifdef __EMSCRIPTEN__
    model->config.num_layers = L = 12; // TODO(avh): Debugging only hack - revert this
#else
    model->config.num_layers = L = model_header[4];
#endif
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);
    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    malloc_and_point_parameters(ctx, model->config, &model->params, model->param_sizes);
    model->params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // transfer to GPU memory
    float* iter = model->params_memory;
    toGPU(ctx, iter, model->params.wte);
    iter += model->param_sizes[0];
    toGPU(ctx, iter, model->params.wpe);
    iter += model->param_sizes[1];
    for (int l = 0; l < L; l++) {
        toGPU(ctx, iter, model->params.ln1w[l]);
        iter += model->param_sizes[2]/L;
        toGPU(ctx, iter, model->params.ln1b[l]);
        iter += model->param_sizes[3]/L;
        toGPU(ctx, iter, model->params.qkvw[l]);
        iter += model->param_sizes[4]/L;
        toGPU(ctx, iter, model->params.qkvb[l]);
        iter += model->param_sizes[5]/L;
        toGPU(ctx, iter, model->params.attprojw[l]);
        iter += model->param_sizes[6]/L;
        toGPU(ctx, iter, model->params.attprojb[l]);
        iter += model->param_sizes[7]/L;
        toGPU(ctx, iter, model->params.ln2w[l]);
        iter += model->param_sizes[8]/L;
        toGPU(ctx, iter, model->params.ln2b[l]);
        iter += model->param_sizes[9]/L;
        toGPU(ctx, iter, model->params.fcw[l]);
        iter += model->param_sizes[10]/L;
        toGPU(ctx, iter, model->params.fcb[l]);
        iter += model->param_sizes[11]/L;
        toGPU(ctx, iter, model->params.fcprojw[l]);
        iter += model->param_sizes[12]/L;
        toGPU(ctx, iter, model->params.fcprojb[l]);
        iter += model->param_sizes[13]/L;
    }
    toGPU(ctx, iter, model->params.lnfw);
    iter += model->param_sizes[14];
    toGPU(ctx, iter, model->params.lnfb);
    iter += model->param_sizes[15];
    

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
    model->mean_loss_buffer = NULL;
    model->probs_buffer = NULL;
    model->backward_enabled = false;

    printf("Model build complete\n");

}


void gpt2_forward(Context& ctx, GPT2 *model, Tensor& inputs, Tensor& targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // // validate inputs, all indices must be in the range [0, V)
    // for(int i = 0; i < B * T; i++) {
    //     assert(0 <= inputs[i] && inputs[i] < V);
    //     if (targets != NULL) {
    //         assert(0 <= targets[i] && targets[i] < V);
    //     }
    // }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        model->mean_loss_buffer = (float*)mallocCheck(sizeof(float) * model->batch_size * model->seq_len);
        model->probs_buffer =  (float*)mallocCheck(sizeof(float) * model->batch_size * model->seq_len * Vp);

        // TODO(avh): this is just a resource test for now, eventually deprecate CPU allocations
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        printf("Allocating %.2f MB for activations\n", num_activations * sizeof(float) / (1024.0f * 1024.0f));
        malloc_and_point_activations(ctx, model->config, &model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        //model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        //model->targets = (int*)mallocCheck(B * T * sizeof(int)); // might be unused if we never have targets but it's small
        model->inputs = createTensor(ctx, Shape{B * T}, ki32);
        model->targets = createTensor(ctx, Shape{B * T}, ki32);
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }
    // create all kernels ahead of time
    if (model->kernels.encoder_forward == nullptr) {
        printf("Creating Kernels\n");
        Kernels& kernels = model->kernels;
        kernels.layernorm_forward.resize(L);
        kernels.layernorm1_backward.resize(L);
        kernels.qkv_projection_forward.resize(L);
        kernels.qkv_projection_backward.resize(L);
        kernels.attention_forward.resize(L);
        kernels.attention_backward.resize(L);
        kernels.attention_projection_forward.resize(L);
        kernels.attention_projection_backward.resize(L);
        kernels.residual_forward.resize(L);
        kernels.residual2_forward.resize(L);
        kernels.residual2_backward.resize(L);
        kernels.ff_up_forward.resize(L);
        kernels.ff_up_backward.resize(L);
        kernels.gelu_forward.resize(L);
        kernels.gelu_backward.resize(L);
        kernels.ff_down_forward.resize(L);
        kernels.ff_down_backward.resize(L);
        for (int l = 0; l < L; ++l) {
            kernels.layernorm_forward[l] = layernorm_forward(ctx, model->acts.ln1[l], model->acts.ln1_mean[l], model->acts.ln1_rstd[l],
                                                            /*input=*/ model->acts.residual3[l], /*weight=*/ model->params.ln1w[l], /*bias=*/ model->params.ln1b[l],
                                                            B, T, C);
            kernels.qkv_projection_forward[l] = matmul_forward(ctx, model->acts.qkv[l], model->acts.ln1[l], model->params.qkvw[l], model->params.qkvb[l], B, T, C, 3*C);
            kernels.attention_forward[l] = attention_forward(ctx, model->acts.atty[l], model->acts.preatt[l], model->acts.att[l], model->acts.qkv[l], B, T, C, NH);
            kernels.attention_projection_forward[l] = matmul_forward(ctx, model->acts.attproj[l], model->acts.atty[l], model->params.attprojw[l], model->params.attprojb[l], B, T, C, C);
            kernels.residual_forward[l] = residual_forward(ctx, model->acts.residual2[l], model->acts.residual3[l], model->acts.attproj[l], B*T*C);
            kernels.ff_up_forward[l] = matmul_forward(ctx, model->acts.fch[l], model->acts.ln2[l], model->params.fcw[l], model->params.fcb[l], B, T, C, 4*C);
            kernels.gelu_forward[l] = gelu_forward(ctx, model->acts.fch_gelu[l], model->acts.fch[l], B*T*4*C);
            kernels.ff_down_forward[l] = matmul_forward(ctx, model->acts.fcproj[l], model->acts.fch_gelu[l], model->params.fcw[l], model->params.fcb[l], B, T, 4*C, C);
            kernels.residual2_forward[l] = residual_forward(ctx, model->acts.residual3[l], model->acts.residual2[l], model->acts.fcproj[l], B*T*C);
        }
        kernels.crossentropy_forward = crossentropy_forward(ctx, model->acts.losses, model->acts.probs, targets, B, T, Vp);
        
        kernels.encoder_forward = encoder_forward(ctx, model->acts.encoded, inputs, model->params.wte, model->params.wpe, B, T, C); // encoding goes into residual[0]
        if(model->backward_enabled)
          kernels.encoder_backward = encoder_backward(ctx, model->params.wte, model->params.wpe, model->acts.encoded, inputs, B, T, C);
        kernels.layernorm_final_forward = layernorm_forward(ctx, model->acts.lnf, model->acts.lnf_mean, model->acts.lnf_rstd,
                                                        /*input=*/ model->acts.residual3[L-1], /*weight=*/ model->params.lnfw, /*bias=*/ model->params.lnfb,
                                                        B, T, C);
        Tensor nullTensor = createTensor(ctx, Shape{1}, kf32);
        model->nullTensor = nullTensor;
        kernels.matmul_final_forward = matmul_forward(ctx, model->acts.logits, model->acts.lnf, model->params.wte, nullTensor, B, T, C, Vp);
        kernels.softmax_final_forward = softmax_forward(ctx, model->acts.probs, model->acts.logits, B, T, V, Vp);
        if(model->backward_enabled)
          kernels.crossentropy_softmax_backward = crossentropy_softmax_backward(ctx, model->acts.logits, model->acts.losses, model->acts.probs, targets, B, T, V, Vp);
        if(model->backward_enabled)
          kernels.matmul_final_backward = matmul_backward(ctx, model->acts.lnf, model->params.wte, nullTensor, model->acts.logits,
                                                          model->acts.lnf, model->params.wte, B, T, C, Vp);
        if(model->backward_enabled)
          kernels.layernorm_final_backward = layernorm_backward(ctx, model->acts.residual3[L-1], model->params.lnfw, model->params.lnfb,
                                                                model->acts.lnf, model->acts.residual3[L-1], model->params.lnfw,
                                                                model->acts.lnf_mean, model->acts.lnf_rstd, B, T, C);
        printf("Created Kernels\n");
    }

    printf("Cache inputs/targets\n");
    printf("Forward pass\n");
    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    printf("Encoding\n");
    //printf("inputs[0] = %d\n", inputs[0]);
    // encoder_forward(ctx, acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.encoder_forward, promise);
        wait(ctx, future);
    }
    for (int l = 0; l < L; l++) {
      printf("Forward Pass Layer %d\n", l);

        // now do the forward pass
        printf("  [Forward] : LayerNorm1\n");
        // layernorm_forward(ctx, l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.layernorm_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : QKV Projection\n");
        // matmul_forward(ctx, l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.qkv_projection_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : Attention\n");
        // attention_forward(ctx, l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.attention_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : Attention Projection\n");
        // matmul_forward(ctx, l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.attention_projection_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : Residual1\n");
        // residual_forward(ctx, l_residual2, residual, l_attproj, B*T*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.residual_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : LayerNorm2\n");
        // layernorm_forward(ctx, l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.layernorm_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : FF Up\n");
        // matmul_forward(ctx, l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.ff_up_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : GELU\n");
        // gelu_forward(ctx, l_fch_gelu, l_fch, B*T*4*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.gelu_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : FF Down\n");
        // matmul_forward(ctx, l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.ff_down_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Forward] : Residual2\n");
        // residual_forward(ctx, l_residual3, l_residual2, l_fcproj, B*T*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.residual2_forward[l], promise);
            wait(ctx, future);
        }
    }
    //    residual = acts.residual3.data() + (L-1) * B * T * C; // last residual is in residual3
    // layernorm_forward(ctx, acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.layernorm_final_forward, promise);
        wait(ctx, future);
    }
    // matmul_forward(ctx, acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.matmul_final_forward, promise);
        wait(ctx, future);
    }
    // softmax_forward(ctx, acts.probs, acts.logits, B, T, V, Vp);
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.softmax_final_forward, promise);
        wait(ctx, future);
    }

    printf("Crossentropy\n");
    // also forward the cross-entropy loss function if we have the targets
    // When targets's shape is (1), it means we don't have targets
    if (targets.shape[0] != 1) {
        // crossentropy_forward(ctx, model->acts.losses, model->acts.probs, targets, B, T, Vp);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.crossentropy_forward, promise);
            wait(ctx, future);
        }
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        toCPU(ctx, model->acts.losses, model->mean_loss_buffer, B*T * sizeof(float));
        for (int i=0; i<B*T; i++) { mean_loss += model->mean_loss_buffer[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
    printf("Forward pass done\n");
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void gpt2_backward(Context& ctx, GPT2 *model) {
    printf("Backward pass\n");

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        printf("Allocating %.2f MB for gradients\n", model->num_parameters * sizeof(float) / (1024.0f * 1024.0f));
        malloc_and_point_parameters(ctx, model->config, &model->grads, model->param_sizes);
        malloc_and_point_activations(ctx, model->config, &model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { model->mean_loss_buffer[i] = dloss_mean; }
    toGPU(ctx, model->mean_loss_buffer, model->acts.losses);
    //toGPU(ctx, grads_acts.losses.data, model->acts_.data[22]);

    // crossentropy_softmax_backward(ctx, grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.crossentropy_softmax_backward, promise);
        wait(ctx, future);
    }
    // matmul_backward(ctx, grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.matmul_final_backward, promise);
        wait(ctx, future);
    }
    // layernorm_backward(ctx, dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.layernorm_final_backward, promise);
        wait(ctx, future);
    }

    for (int l = L-1; l >= 0; l--) {
        printf("Backward Pass Layer %d\n", l);
        // backprop this layer
        printf("  [Backward] : Residual2\n");
        // residual_backward(ctx, dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.residual2_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : FF Down \n");
        // matmul_backward(ctx, dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.ff_down_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : GELU\n");
        // gelu_backward(ctx, dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.gelu_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : FF Up\n");
        // matmul_backward(ctx, dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.ff_up_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : LayerNorm2\n");
        // layernorm_backward(ctx, dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.layernorm2_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : Residual1\n");
        // residual_backward(ctx, dresidual, dl_attproj, dl_residual2, B*T*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.residual_forward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : Attention Projection\n");
        // matmul_backward(ctx, dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.attention_projection_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : Attention\n");
        // attention_backward(ctx, dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.attention_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : QKV Projection\n");
        // matmul_backward(ctx, dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.qkv_projection_backward[l], promise);
            wait(ctx, future);
        }
        printf("  [Backward] : LayerNorm1\n");
        // layernorm_backward(ctx, dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            dispatchKernel(ctx, model->kernels.layernorm1_backward[l], promise);
            wait(ctx, future);
        }
    }
    // encoder_backward(ctx, grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, model->kernels.encoder_backward, promise);
        wait(ctx, future);
    }
    // toCPU(ctx, model->params_.data[0], model->grads.wte.data, model->param_sizes[0] * sizeof(float));
    // toCPU(ctx, model->params_.data[1], model->grads.wpe.data, model->param_sizes[1] * sizeof(float));
}

void gpt2_update(Context& ctx, GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    // Copy the parameters to the CPU
    float* iter = model->params_memory;
    toCPU(ctx, model->params.wte, iter, model->param_sizes[0] * sizeof(float));
    iter += model->param_sizes[0];
    toCPU(ctx, model->params.wpe, iter, model->param_sizes[1] * sizeof(float));
    iter += model->param_sizes[1];
    size_t L = model->config.num_layers;
    for (int l = 0; l < L; l++) {
        toCPU(ctx, model->params.ln1w[l], iter, model->param_sizes[2]/L * sizeof(float));
        iter += model->param_sizes[2]/L;
        toCPU(ctx, model->params.ln1b[l], iter, model->param_sizes[3]/L * sizeof(float));
        iter += model->param_sizes[3]/L;
        toCPU(ctx, model->params.qkvw[l], iter, model->param_sizes[4]/L * sizeof(float));
        iter += model->param_sizes[4]/L;
        toCPU(ctx, model->params.qkvb[l], iter, model->param_sizes[5]/L * sizeof(float));
        iter += model->param_sizes[5]/L;
        toCPU(ctx, model->params.attprojw[l], iter, model->param_sizes[6]/L * sizeof(float));
        iter += model->param_sizes[6]/L;
        toCPU(ctx, model->params.attprojb[l], iter, model->param_sizes[7]/L * sizeof(float));
        iter += model->param_sizes[7]/L;
        toCPU(ctx, model->params.ln2w[l], iter, model->param_sizes[8]/L * sizeof(float));
        iter += model->param_sizes[8]/L;
        toCPU(ctx, model->params.ln2b[l], iter, model->param_sizes[9]/L * sizeof(float));
        iter += model->param_sizes[9]/L;
        toCPU(ctx, model->params.fcw[l], iter, model->param_sizes[10]/L * sizeof(float));
        iter += model->param_sizes[10]/L;
        toCPU(ctx, model->params.fcb[l], iter, model->param_sizes[11]/L * sizeof(float));
        iter += model->param_sizes[11]/L;
        toCPU(ctx, model->params.fcprojw[l], iter, model->param_sizes[12]/L * sizeof(float));
        iter += model->param_sizes[12]/L;
        toCPU(ctx, model->params.fcprojb[l], iter, model->param_sizes[13]/L * sizeof(float));
        iter += model->param_sizes[13]/L;
    }
    toCPU(ctx, model->params.lnfw, iter, model->param_sizes[14] * sizeof(float));
    iter += model->param_sizes[14];
    toCPU(ctx, model->params.lnfb, iter, model->param_sizes[15] * sizeof(float));
    iter += model->param_sizes[15];
    

    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
    // toGPU(ctx, model->params_memory, model->params_.data[0]);
    // toGPU(ctx, model->params_memory + model->param_sizes[0], model->params_.data[1]);
    iter = model->params_memory;
    toGPU(ctx, iter, model->params.wte);
    iter += model->param_sizes[0];
    toGPU(ctx, iter, model->params.wpe);
    iter += model->param_sizes[1];
    for (int l = 0; l < L; l++) {
        toGPU(ctx, iter, model->params.ln1w[l]);
        iter += model->param_sizes[2]/L;
        toGPU(ctx, iter, model->params.ln1b[l]);
        iter += model->param_sizes[3]/L;
        toGPU(ctx, iter, model->params.qkvw[l]);
        iter += model->param_sizes[4]/L;
        toGPU(ctx, iter, model->params.qkvb[l]);
        iter += model->param_sizes[5]/L;
        toGPU(ctx, iter, model->params.attprojw[l]);
        iter += model->param_sizes[6]/L;
        toGPU(ctx, iter, model->params.attprojb[l]);
        iter += model->param_sizes[7]/L;
        toGPU(ctx, iter, model->params.ln2w[l]);
        iter += model->param_sizes[8]/L;
        toGPU(ctx, iter, model->params.ln2b[l]);
        iter += model->param_sizes[9]/L;
        toGPU(ctx, iter, model->params.fcw[l]);
        iter += model->param_sizes[10]/L;
        toGPU(ctx, iter, model->params.fcb[l]);
        iter += model->param_sizes[11]/L;
        toGPU(ctx, iter, model->params.fcprojw[l]);
        iter += model->param_sizes[12]/L;
        toGPU(ctx, iter, model->params.fcprojb[l]);
        iter += model->param_sizes[13]/L;
    }
    toGPU(ctx, iter, model->params.lnfw);
    iter += model->param_sizes[14];
    toGPU(ctx, iter, model->params.lnfb);
    iter += model->param_sizes[15];
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    //    free(model->inputs);
    //    free(model->targets);
    free(model->mean_loss_buffer);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main() {

    setLogLevel(kWarn);

    printf("Creating GPU context\n");
    WGPURequiredLimits requiredLimits = LIMITS_BUFFER_SIZE_1GB;
    gpu::Context ctx = gpu::createContext({}, {}, {
        .requiredLimits = &requiredLimits
    });
    
   // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(ctx, &model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    constexpr int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    constexpr int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    // int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    Tensor inputs = createTensor(ctx, Shape{B, T}, ki32);
    Tensor targets = createTensor(ctx, Shape{B, T}, ki32);
    Tensor gen_tokens = createTensor(ctx, Shape{B, T}, ki32);
    int* gen_tokens_cpu = (int*)mallocCheck(B * T * sizeof(int));
    printf("Starting training\n");
    for (int step = 0; step <= 40; step++) {
        printf("Step %d\n", step);

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                toGPU(ctx, val_loader.inputs, inputs);
                toGPU(ctx, val_loader.targets, targets);
                gpt2_forward(ctx, &model, inputs, targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens_cpu[i] = tokenizer.eot_token;
            }
            toGPU(ctx, gen_tokens_cpu, gen_tokens);
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(ctx, &model, gen_tokens, model.nullTensor, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                toCPU(ctx, model.acts.probs, model.probs_buffer, B * T * model.config.padded_vocab_size * sizeof(float));
                float* probs = model.probs_buffer + (t-1) * model.config.padded_vocab_size;

                float coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens_cpu[t] = next_token;
                toGPU(ctx, gen_tokens_cpu, gen_tokens);
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        toGPU(ctx, train_loader.inputs, inputs);
        toGPU(ctx, train_loader.targets, targets);
        gpt2_forward(ctx, &model, inputs, targets, B, T);
        if (model.backward_enabled) {
            gpt2_zero_grad(&model);
            gpt2_backward(ctx, &model);
            gpt2_update(ctx, &model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    // free(gen_tokens);
    return 0;
}
#endif
