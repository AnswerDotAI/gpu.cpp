CXX=clang++
CC=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif

# ASYNCIFY allows emscripten to sleep
EMFLAGS=-std=c++17 -I$(GPUCPP) -I$(GPUCPP)/third_party/headers/wasm -I. -Iunittest_llmc -I$(GPUCPP)/third_party/llm.c -s USE_WEBGPU=1 -s ASYNCIFY=1 -s STACK_SIZE=100000 -s MEMORY64=1 -s ALLOW_MEMORY_GROWTH=1
CXXFLAGS=-std=c++17 -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -I. -Iunittest_llmc
CFLAGS=-Ofast -march=native -I. -Iunittest_llmc
# CFLAGS=-O2 -march=native -I. -Iunittest_llmc

LDFLAGS=$(STDLIB) -L$(GPUCPP)/third_party/lib -ldl -lwebgpu_dawn -fsanitize=address
FLAGS=$(CXXFLAGS) $(LDFLAGS)

ifeq ($(shell [ -d /opt/homebrew/opt/libomp/lib ] && echo "exists"), exists)
  CFLAGS += -Xclang -fopenmp -DOMP -I/opt/homebrew/opt/libomp/include
  LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
  $(info ✓ OpenMP found)
else
  $(info ✗ OpenMP not found)
endif

default: run-native

build/reduce: reduce.cpp kernels.h
	$(CC) $(CFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	$(LIBSPEC) && build/reduce

run_llm.c: ./build/test_gpt2 dawnlib
	$(LIBSPEC) && $<

run_llm.c_with_metal_profiler: ./build/test_gpt2_with_metal_profiler dawnlib
	$(LIBSPEC) && export METAL_CAPTURE_ENABLED=1 && $<

run_llm.c_with_time_profiler: ./build/test_gpt2_with_metal_profiler dawnlib
	$(LIBSPEC) && xcrun xctrace record --template 'Time Profiler' --launch -- $<

run_llm.c_train: ./build/train_gpt2 dawnlib
	if [ ! -d dev ] ; then ln -s $(GPUCPP)/third_party/llm.c/dev ; fi
	if [ ! -f gpt2_tokenizer.bin ] ; then ln -s $(GPUCPP)/third_party/llm.c/gpt2_tokenizer.bin ; fi
	$(LIBSPEC) && $<

llm.c:
	# if [ ! -d llm.c ]; then git clone git@github.com:karpathy/llm.c.git ; fi
	ln -s $(GPUCPP)/third_party/llm.c

gpt2_124M.bin: llm.c
	if [ ! -f $@ ]; then ./llm.c/dev/download_starter_pack.sh ; \
          ln -s ./llm.c/gpt2_124M.bin ; \
          ln -s ./llm.c/gpt2_124M_debug_state.bin ; \
					ln -s ./llm.c/gpt2_tokenizer.bin ; \
	fi

define preprocess_file
	sed -i -e 's/int main(/int MAIN(/g' llm.c/test_gpt2.c
	sed -i -e 's/int main(/int MAIN(/g' llm.c/train_gpt2.c
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
	grep -q "^#include \"unittest_kernels.h\"" llm.c/train_gpt2.c || \
		printf '1i\n#include "unittest_kernels.h"\n.\nw\nq\n' | ed -s llm.c/train_gpt2.c
endef

build/test_gpt2: llm.c build/unittest_kernels.o gpt2_124M.bin
	mkdir -p build
	$(call preprocess_file)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ llm.c/test_gpt2.c build/unittest_kernels.o

build/test_gpt2_with_metal_profiler: llm.c build/unittest_kernels.o gpt2_124M.bin
	mkdir -p build
	$(call preprocess_file)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ llm.c/test_gpt2.c build/unittest_kernels.o -I$(GPUCPP) $(GPUCPP)/experimental/profiler/metal.mm -framework metal -framework Foundation -DMETAL_PROFILER -g
	install_name_tool -change @rpath/libwebgpu_dawn.dylib $(GPUCPP)/third_party/lib/libwebgpu_dawn.dylib $@

build/train_gpt2: llm.c build/unittest_kernels.o gpt2_124M.bin
	mkdir -p build
	$(call preprocess_file)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ llm.c/train_gpt2.c build/unittest_kernels.o

build/ops.o: ops.cpp ops.hpp kernels.h llm.c
	mkdir -p build && $(CXX) $(CXXFLAGS) -c -o $@ $<

build/gpt2_webgpu: llm.c gpt2_124M.bin llm.c gpt2_webgpu.cpp ops.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -Illm.c $(LDFLAGS) -o $@ gpt2_webgpu.cpp ops.cpp

build/gpt2_webgpu_aot: llm.c gpt2_124M.bin llm.c gpt2_webgpu_aot.cpp ops_aot.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -Illm.c $(LDFLAGS) -o $@ gpt2_webgpu_aot.cpp ops_aot.cpp

build/gpt2_webgpu.html: check-emsdk gpt2_webgpu.cpp term.html llm.c
	em++ gpt2_webgpu.cpp ops.cpp \
		--preload-file gpt2_tokenizer.bin@/gpt2_tokenizer.bin \
		--preload-file gpt2_124M.bin@/gpt2_124M.bin \
		--preload-file gpt2_124M_debug_state.bin@/gpt2_124M_debug_state.bin \
		--preload-file llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin@dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
		--preload-file llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin@dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
		-o build/gpt2_webgpu.html \
		$(EMFLAGS) \
		--shell-file term.html \

watch-web:
	ls *.cpp *.c *.hpp *.h | entr -s make build/gpt2_webgpu.html

watch-native:
	ls *.cpp *.c *.hpp *.h | entr -s "rm -f build/gpt2_webgpu && rm -f build/ops.o && make build/gpt2_webgpu"

run-native: build/gpt2_webgpu_aot
	. $(GPUCPP)/source && ./build/gpt2_webgpu_aot

# server: build/train_gpt2.html build/test_gpt2.html build/gpt2_gpucpp.html
server: build/gpt2_webgpu.html
	@echo "\n┌───────────────────────────────────────────────────────────────────────────────────┐"
	@echo   "│  Open http://localhost:8000/build/run.html in your browser to see the output.     │"
	@echo	  "│                                                                                   │"
	@echo	  "│  Press Ctrl+C to stop the server.                                                 │"
	@echo   "└───────────────────────────────────────────────────────────────────────────────────┘\n\n"
	python3 -m http.server --directory .

build/unittest_kernels.o: unittest_llmc/unittest_kernels.cpp unittest_llmc/unittest_kernels.h kernels.h
	mkdir -p build && $(CXX) $(CXXFLAGS) -DNDEBUG -c -o $@ $<

dawnlib: $(if $(wildcard $(GPUCPP)/third_party/lib/libwebgpu_dawn.so $(GPUCPP)/third_party/lib/libwebgpu_dawn.dylib),,run_setup)

run_setup: check-python
	cd $(GPUCPP) && python3 setup.py

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python needs to be installed and in your path."; exit 1; } 

check-emsdk:
	@which em++ > /dev/null || (echo "emsdk not found. Please install emsdk and run 'source emsdk_env.sh' in the emsdk directory." && exit 1)
