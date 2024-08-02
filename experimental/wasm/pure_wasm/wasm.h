#ifndef WASM_H
#define WASM_H

// #define WASM_IMPORT __attribute__((import_module("env"),
// import_name("memory"))) #define WASM_IMPORT __attribute__((used))
// __attribute__((visibility("default")))

extern "C" {

// these are normally defined in stdint.h, but we can't include that in wasm
typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long size_t;

// Opaque handles to js shim objects
typedef struct Shape Shape;
typedef struct Array Array;
typedef struct Tensor Tensor;
typedef struct TensorView TensorView;
typedef struct Bindings Bindings;
typedef struct Context Context;
typedef struct KernelCode KernelCode;
typedef struct Kernel Kernel;

// Enum to match JavaScript NumType
typedef enum { kf16, kf32 } NumType;

// Function declarations that will be implemented in JavaScript

Shape *createShape(int32_t *dims, int32_t rank);
void destroyShape(Shape *shape);

Array *createArray(uint64_t bufferPtr, uint32_t usage, uint64_t size);
void destroyArray(Array *array);

Tensor *createTensor(Array *data, Shape *shape);
void destroyTensor(Tensor *tensor);

TensorView *createTensorView(Tensor *data, uint64_t offset, uint64_t span);
void destroyTensorView(TensorView *view);

Bindings *createBindings(Tensor **tensors, int32_t count);
void destroyBindings(Bindings *bindings);

Context *createContext();
void destroyContext(Context *ctx);

KernelCode *createKernelCode(const char *data, Shape *workgroupSize,
                             NumType precision);
void destroyKernelCode(KernelCode *code);

Kernel *createKernel(Context *ctx, KernelCode *code, Bindings *dataBindings,
                     Shape *nWorkgroups, void *params);
void destroyKernel(Kernel *kernel);

uint64_t size(Shape *shape);
uint64_t sizeBytes(NumType type);

char *toString(Shape *shape);
char *toStringInt(int32_t value);
char *toStringNumType(NumType type);

void replaceAll(char *str, const char *from, const char *to);

int32_t cdiv(int32_t n, int32_t d);
Shape *cdivShape(Shape *total, Shape *group);

Tensor *createTensorImpl(Context *ctx, Shape *shape, NumType dtype);

void toGPU(Context *ctx, float *data, Tensor *tensor);
void toCPU(Context *ctx, Tensor *tensor, float *data);

void dispatchKernel(Context *ctx, Kernel *kernel);

void resetCommandBuffer(Context *ctx, Kernel *kernel);

uint8_t *memory;

void jsLOG(uint8_t *messagePtr);

int simpleTest();

} // extern "C"

// Simple bump allocator for now

uint32_t  kMemPtr = 0;

uint8_t* wasmMalloc(size_t size) {
    uint8_t* ptr = &memory[kMemPtr];
    kMemPtr += size;
    return ptr;
}

size_t strlen(const char* str) {
    size_t len = 0;
    while (str[len]) {
        len++;
    }
    return len;
}

void LOG(const char* message) {
    size_t len = strlen(message);
    uint8_t* start = (wasmMalloc(len));
    uint8_t* dest = start;
    size_t index = 0;
    while (*message) {
      *dest = *message;
      dest++;
      message++;
    }
    jsLOG(start);
}

#endif // WASM_H
