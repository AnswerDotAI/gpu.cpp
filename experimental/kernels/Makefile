CXX=clang++
CC=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
TARGET=test_kernels
ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif

CXXFLAGS=-std=c++17 -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -I. -Iunittest_llmc
CFLAGS=-Ofast -march=native -I. -Iunittest_llmc

LDFLAGS=$(STDLIB) -L$(GPUCPP)/third_party/lib -ldl -ldawn
FLAGS=$(CXXFLAGS) $(LDFLAGS)

ifeq ($(shell [ -d /opt/homebrew/opt/libomp/lib ] && echo "exists"), exists)
  CFLAGS += -Xclang -fopenmp -DOMP -I/opt/homebrew/opt/libomp/include
  LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
  $(info ✓ OpenMP found)
else
  $(info ✗ OpenMP not found)
endif

run: ./build/$(TARGET) dawnlib
	$(LIBSPEC) && ./build/$(TARGET)

run_llm.c: ./build/test_gpt2 dawnlib
	$(LIBSPEC) && $<

run_llm.c_train: ./build/train_gpt2 dawnlib
	if [ ! -d dev ] ; then ln -s ./llm.c/dev ; fi
	if [ ! -f gpt2_tokenizer.bin ] ; then ln -s ./llm.c/gpt2_tokenizer.bin ; fi
	$(LIBSPEC) && $<

llm.c:
	if [ ! -d llm.c ]; then git clone git@github.com:karpathy/llm.c.git ; fi

gpt2_124M.bin: llm.c
	if [ ! -f $@ ]; then ./llm.c/dev/download_starter_pack.sh ; \
          ln -s ./llm.c/gpt2_124M.bin ; \
          ln -s ./llm.c/gpt2_124M_debug_state.bin ; \
        fi

build/test_gpt2: llm.c build/unittest_kernels.o gpt2_124M.bin
	mkdir -p build
	sed -i -e 's/void encoder_forward(/void ENCODER_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void layernorm_forward(/void LAYERNORM_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void matmul_forward(/void MATMUL_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void attention_forward(/void ATTENTION_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void gelu_forward(/void GELU_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void residual_forward(/void RESIDUAL_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void softmax_forward(/void SOFTMAX_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void crossentropy_forward(/void CROSSENTROPY_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void encoder_backward(/void ENCODER_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void layernorm_backward(/void LAYERNORM_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void matmul_backward(/void MATMUL_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void attention_backward(/void ATTENTION_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void gelu_backward(/void GELU_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void residual_backward(/void RESIDUAL_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void crossentropy_softmax_backward(/void CROSSENTROPY_SOFTMAX_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	grep -q "^#include \"unittest_kernels.h\"" llm.c/train_gpt2.c || sed -i '1i#include \"unittest_kernels.h\"' llm.c/train_gpt2.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ llm.c/test_gpt2.c build/unittest_kernels.o

build/train_gpt2: llm.c build/unittest_kernels.o gpt2_124M.bin
	mkdir -p build
	sed -i -e 's/void encoder_forward(/void ENCODER_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void layernorm_forward(/void LAYERNORM_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void matmul_forward(/void MATMUL_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void attention_forward(/void ATTENTION_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void gelu_forward(/void GELU_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void residual_forward(/void RESIDUAL_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void softmax_forward(/void SOFTMAX_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void crossentropy_forward(/void CROSSENTROPY_FORWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void encoder_backward(/void ENCODER_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void layernorm_backward(/void LAYERNORM_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void matmul_backward(/void MATMUL_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void attention_backward(/void ATTENTION_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void gelu_backward(/void GELU_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void residual_backward(/void RESIDUAL_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	sed -i -e 's/void crossentropy_softmax_backward(/void CROSSENTROPY_SOFTMAX_BACKWARD_CPU(/g' llm.c/train_gpt2.c
	grep -q "^#include \"unittest_kernels.h\"" llm.c/train_gpt2.c || sed -i '1i#include \"unittest_kernels.h\"' llm.c/train_gpt2.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ llm.c/train_gpt2.c build/unittest_kernels.o

build/unittest_kernels.o: unittest_llmc/unittest_kernels.cpp unittest_llmc/unittest_kernels.h kernels.h
	mkdir -p build && $(CXX) $(CXXFLAGS) -DNDEBUG -c -o $@ $<

dawnlib: $(if $(wildcard $(GPUCPP)/third_party/lib/libdawn.so $(GPUCPP)/third_party/lib/libdawn.dylib),,run_setup)

run_setup: check-python
	cd $(GPUCPP) && python3 setup.py

build/$(TARGET): run.cpp
	mkdir -p build && $(CXX) $(FLAGS) -DNDEBUG -o $@ $<

debug: run.cpp
	mkdir -p build && $(CXX) $(FLAGS) -g -o ./build/$(TARGET) $<

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python needs to be installed and in your path."; exit 1; } 
