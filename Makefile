NUM_JOBS=$(shell nproc)
CXX=clang++

.PHONY: default examples/hello_world/build/hello_world tests libgpu debug build check-entr check-clang clean-build clean all watch-tests

GPUCPP ?= $(PWD)
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source

default: examples/hello_world/build/hello_world

examples/hello_world/build/hello_world: check-clang dawnlib examples/hello_world/run.cpp check-linux-vulkan
	$(LIBSPEC) && cd examples/hello_world && make build/hello_world && ./build/hello_world

dawnlib: $(if $(wildcard third_party/lib/libdawn.so third_party/lib/libdawn.dylib),,run_setup)

run_setup: check-python
	python3 setup.py

all: dawnlib check-clang check-linux-vulkan
	cd examples/gpu_puzzles && make build/gpu_puzzles
	cd examples/hello_world && make build/hello_world
	cd examples/matmul && make build/mm
	cd examples/physics && make build/physics
	cd examples/render && make build/render

docs:
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
	rm -f third_party/lib/libdawn.so third_party/lib/libdawn.dylib

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
	rm -rf examples/gpu_puzzles/build/*
	rm -rf examples/hello_world/build/*
	rm -rf examples/matmul/build/mm
	rm -rf examples/physics/build/*
	rm -rf examples/render/build/*

clean-all:
	read -r -p "This will delete the contents of build/* and third_party/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/* third_party/fetchcontent/* third_party/gpu-build third_party/gpu-subbuild third_party/gpu-src third_party/lib/libdawn.so third_party/lib/libdawn.dylib

################################################################################
# Checks
################################################################################

# check for the existence of clang++ and cmake
check-clang:
	@command -v clang++ >/dev/null 2>&1 || { echo >&2 "Please install clang++ with 'sudo apt-get install clang' or 'brew install llvm'"; exit 1; }

check-entr:
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }

check-cmake:
	@command -v cmake >/dev/null 2>&1 || { echo >&2 "Please install cmake with 'sudo apt-get install cmake' or 'brew install cmake'"; exit 1; }

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python needs to be installed and in your path."; exit 1; } 

check-linux-vulkan:
	@echo "Checking system type and Vulkan availability..."
	@if [ "$$(uname)" = "Linux" ]; then \
	    if command -v vulkaninfo >/dev/null 2>&1; then \
	        echo "Vulkan is installed."; \
	        vulkaninfo; \
	    else \
        	echo "Vulkan is not installed. Please install Vulkan drivers to continue. On Debian / Ubuntu: sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools"; \
	        exit 1; \
	    fi \
	else \
	    echo "Non-Linux system detected. Skipping Vulkan check."; \
	fi
