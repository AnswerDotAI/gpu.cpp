NUM_JOBS=$(shell nproc)
CXX=clang++

.PHONY: default examples/hello_world/build/hello_world tests libgpu debug build check-clang clean-build clean all watch-tests docs

GPUCPP ?= $(PWD)
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
INCLUDES ?= -I$(GPUCPP) -I$(GPUCPP)/third_party/headers
ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif

default: examples/hello_world/build/hello_world

pch:
	mkdir -p build && $(CXX) -std=c++17 $(INCLUDES) -x c++-header gpu.hpp -o build/gpu.hpp.pch

# TODO(avh): change extension based on platform
# Get the current OS name
OS = $(shell uname | tr -d '\n')
# Set the specific variables for each platform
LIB_PATH ?= /usr/lib
HEADER_PATH ?= /usr/include
ifeq ($(OS), Linux)
OS_TYPE ?= Linux
GPU_CPP_LIB_NAME ?= libgpucpp.so
DAWN_LIB_NAME ?= libwebgpu_dawn.so
else ifeq ($(OS), Darwin)
OS_TYPE ?= macOS
GPU_CPP_LIB_NAME ?= libgpucpp.dylib
DAWN_LIB_NAME ?= libwebgpu_dawn.dylib
else
OS_TYPE ?= unknown
endif

lib: check-clang dawnlib
	mkdir -p build && $(CXX) -std=c++17 $(INCLUDES) -L$(LIBDIR) -lwebgpu_dawn -ldl -shared -fPIC gpu.cpp -o build/$(GPU_CPP_LIB_NAME)
	python3 build.py
	cp third_party/lib/$(DAWN_LIB_NAME) build/

install:
	cp build/$(GPU_CPP_LIB_NAME) $(LIB_PATH)
	cp build/$(DAWN_LIB_NAME) $(LIB_PATH)
	cp build/gpu.hpp $(HEADER_PATH)

uninstall:
	rm $(LIB_PATH)/$(GPU_CPP_LIB_NAME)
	rm $(LIB_PATH)/$(DAWN_LIB_NAME)
	rm $(HEADER_PATH)/gpu.hpp

examples/hello_world/build/hello_world: check-clang dawnlib examples/hello_world/run.cpp check-linux-vulkan
	$(LIBSPEC) && cd examples/hello_world && make build/hello_world && ./build/hello_world

dawnlib: $(if $(wildcard third_party/lib/libwebgpu_dawn.so third_party/lib/libwebgpu_dawn.dylib),,run_setup)

run_setup: check-python
	python3 setup.py

all: dawnlib check-clang check-linux-vulkan lib pch
	cd examples/float16 && make build/float16
	cd examples/gpu_puzzles && make build/gpu_puzzles
	cd examples/hello_world && make build/hello_world
	cd examples/matmul && make build/matmul
	cd examples/physics && make build/physics
	cd examples/render && make build/render
	cd examples/shadertui && make build/shadertui
	cd examples/transpose && make build/transpose

# Test 16-bit floating point type
test-half: dawnlib check-clang
	$(LIBSPEC) && clang++ -std=c++17 $(INCLUDES) numeric_types/half.cpp -L$(LIBDIR) -lwebgpu_dawn -ldl -o build/half && ./build/half

docs: Doxyfile
	doxygen Doxyfile

################################################################################
# cmake targets (optional - precompiled binaries is preferred)
################################################################################

CMAKE_CMD = mkdir -p build && cd build && cmake ..
# Add --trace to see the cmake commands
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX) -DABSL_INTERNAL_AT_LEAST_CXX20=OFF
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON
DEBUG_FLAGS = $(FLAGS) -DDEBUG:BOOL=ON
RELEASE_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=OFF
TARGET_LIB=gpu

libgpu-cmake: check-clang check-cmake
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) gpu

debug-cmake: check-clang check-cmake
	$(CMAKE_CMD) $(DEBUG_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

all-cmake: check-clang check-cmake
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

################################################################################
# Cleanup
################################################################################

clean-dawnlib:
	rm -f third_party/lib/libwebgpu_dawn.so third_party/lib/libwebgpu_dawn.dylib

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
	rm -rf examples/float16/build/*
	rm -rf examples/gpu_puzzles/build/*
	rm -rf examples/hello_world/build/*
	rm -rf examples/matmul/build/matmul
	rm -rf examples/physics/build/*
	rm -rf examples/render/build/*
	rm -rf examples/shadertui/build/*
	rm -rf examples/transpose/build/transpose
	rm -f build/gpu.hpp.pch
	rm -f build/libgpucpp.so
	rm -f build/half

clean-all:
	read -r -p "This will delete the contents of build/* and third_party/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/* third_party/fetchcontent/* third_party/gpu-build third_party/gpu-subbuild third_party/gpu-src third_party/lib/libwebgpu_dawn.so third_party/lib/libwebgpu_dawn.dylib

################################################################################
# Checks
################################################################################

# Check all
check-all: check-os check-clang check-cmake check-python

# check the os
check-os:
ifeq ($(OS_TYPE), unknown)
$(error Unsupported operating system)
endif

# check for the existence of clang++ and cmake
check-clang:
	@command -v clang++ >/dev/null 2>&1 || { echo -e >&2 "Clang++ is not installed. Please install clang++ to continue.\nOn Debian / Ubuntu: 'sudo apt-get install clang' or 'brew install llvm'\nOn Centos: 'sudo yum install clang'"; exit 1; }

check-cmake:
	@command -v cmake >/dev/null 2>&1 || { echo -e >&2 "Cmake is not installed. Please install cmake to continue.\nOn Debian / Ubuntu: 'sudo apt-get install cmake' or 'brew install cmake'\nOn Centos: 'sudo yum install cmake'"; exit 1; }

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo -e >&2 "Python is not installed. Please install python to continue.\nOn Debian / Ubuntu: 'sudo apt-get install python'\nOn Centos: 'sudo yum install python'"; exit 1; } 

check-linux-vulkan:
	@echo "Checking system type and Vulkan availability..."
	@if [ "$$(uname)" = "Linux" ]; then \
	    if command -v vulkaninfo >/dev/null 2>&1; then \
	        echo "Vulkan is installed."; \
	        vulkaninfo; \
	    else \
		echo -e "Vulkan is not installed. Please install Vulkan drivers to continue.\nOn Debian / Ubuntu: 'sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools'.\nOn Centos: 'sudo yum install vulkan vulkan-tools.'"; \
	        exit 1; \
	    fi \
	else \
	    echo "Non-Linux system detected. Skipping Vulkan check."; \
	fi
