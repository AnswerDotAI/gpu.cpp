NUM_JOBS=$(shell nproc)
CXX=clang++
TARGET_DEMO=run_demo
TARGET_TESTS=run_tests
TARGET_LIB=gpu
TARGET_ALL=$(TARGET_DEMO) $(TARGET_TESTS) $(TARGET_LIB)
USE_LOCAL=-DUSE_LOCAL_LIBS=ON

.PHONY: demo tests libgpu debug build check-entr watch-demo watch-tests clean

# Add --trace to see the cmake commands
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX) -DABSL_INTERNAL_AT_LEAST_CXX20=OFF

FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON
DEBUG_FLAGS = $(FLAGS) -DDEBUG:BOOL=ON
RELEASE_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=OFF
LOCAL_FLAGS = -DUSE_LOCAL_LIBS=ON 
CMAKE_CMD = mkdir -p build && cd build && cmake ..

demo: check-dependencies
	$(CMAKE_CMD) $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET_DEMO) && ./$(TARGET_DEMO)

tests: check-dependencies
	$(CMAKE_CMD) $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)

libgpu: check-dependencies
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) gpu

debug: check-dependencies
	$(CMAKE_CMD) $(DEBUG_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

build: check-dependencies
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

watch-tests: check-entr check-dependencies
	$(CMAKE_CMD) $(FASTBUILD_FLAGS) && ls ../utils/test_kernels.cpp ../*.h ../utils/*.h ../nn/*.h | entr -s "rm -f $(TARGET_TESTS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)"

all: build
	cd examples/gpu_puzzles && make
	cd examples/hello_world && make
	cd examples/raymarch && make
	cd examples/webgpu_intro && make

clean-build:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

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

################################################################################
# Experimental targets (not tested / not working)
################################################################################

USE_WGPU=-DWEBGPU_TAG=wgpu
# EMSCRIPTEN_FLAGS = -DIMPLEMENTATION=emscripten -DCMAKE_TOOLCHAIN_FILE=../cmake/emscripten.cmake -DCMAKE_CXX_COMPILER=em++
EMSCRIPTEN_FLAGS = -DIMPLEMENTATION=emscripten -DCMAKE_CXX_COMPILER=em++

debug-wgpu: check-dependencies
	$(CMAKE_CMD) $(DEBUG_FLAGS) $(USE_WGPU) && make -j$(NUM_JOBS) $(TARGET_ALL)

watch-tests-wgpu: check-entr check-dependencies
	# export RUST_TRACE=1
	$(CMAKE_CMD) $(FASTBUILD_FLAGS) $(USE_WGPU) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_TESTS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)"

emscripten: check-dependencies
	$(CMAKE_CMD) $(EMSCRIPTEN_FLAGS) -DIMPLEMENTATION=emscripten && make -j$(NUM_JOBS) $(TARGET_ALL)
