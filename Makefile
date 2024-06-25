NUM_JOBS=$(shell nproc)
CXX=clang++
TARGET_TESTS=run_tests
TARGET_LIB=gpu
TARGET_ALL=$(TARGET_DEMO) $(TARGET_TESTS) $(TARGET_LIB)
USE_LOCAL=-DUSE_LOCAL_LIBS=ON

.PHONY: default examples/hello_world/build/hello_world tests libgpu debug build check-entr check-dependencies clean-build clean all

# Add --trace to see the cmake commands
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX) -DABSL_INTERNAL_AT_LEAST_CXX20=OFF

FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON
DEBUG_FLAGS = $(FLAGS) -DDEBUG:BOOL=ON
RELEASE_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=OFF
LOCAL_FLAGS = -DUSE_LOCAL_LIBS=ON 
CMAKE_CMD = mkdir -p build && cd build && cmake ..
GPUCPP ?= $(PWD)
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source

default: examples/hello_world/build/hello_world

examples/hello_world/build/hello_world: check-dependencies dawnlib examples/hello_world/run.cpp
	$(LIBSPEC) && cd examples/hello_world && make build/hello_world && ./build/hello_world

build/run_tests: check-dependencies dawnlib
	mkdir -p build && $(CXX) -std=c++17 -I$(GPUCPP) -I$(GPUCPP)/utils -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib utils/test_kernels.cpp -ldawn -ldl -o ./build/run_tests && $(LIBSPEC) && ./build/run_tests

tests-cmake: check-dependencies
	$(CMAKE_CMD) $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)

dawnlib: $(if $(wildcard third_party/lib/libdawn.so third_party/lib/libdawn.dylib),,run_setup)

run_setup:
	python setup.py

libgpu-cmake: check-dependencies
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) gpu

debug-cmake: check-dependencies
	$(CMAKE_CMD) $(DEBUG_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

all-cmake: check-dependencies
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

all: build
	cd examples/gpu_puzzles && make build/gpu_puzzles
	cd examples/hello_world && make build/hello_world
	cd examples/render && make build/render
	cd examples/webgpu_intro && make/physics

clean-build:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
	rm -rf examples/hello_world/build/*

clean:
	read -r -p "This will delete the contents of build/* and third_party/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/* third_party/fetchcontent/* third_party/gpu-build third_party/gpu-subbuild third_party/gpu-src

################################################################################
# Checks
################################################################################

# check for the existence of clang++ and cmake
check-dependencies:
	@command -v clang++ >/dev/null 2>&1 || { echo >&2 "Please install clang++ with 'sudo apt-get install clang' or 'brew install llvm'"; exit 1; }
	@command -v cmake >/dev/null 2>&1 || { echo >&2 "Please install cmake with 'sudo apt-get install cmake' or 'brew install cmake'"; exit 1; }

check-entr:
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }

