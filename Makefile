NUM_JOBS=$(shell nproc)
CXX=clang++
TARGET_DEMO=run_demo
TARGET_TESTS=run_tests
TARGET_LIB=gpu
TARGET_ALL=$(TARGET_DEMO) $(TARGET_TESTS) $(TARGET_LIB)
USE_LOCAL=-DUSE_LOCAL_LIBS=ON
USE_WGPU=-DWEBGPU_TAG=wgpu

.PHONY: demo tests libgpu debug build check-entr watch-demo watch-tests clean

FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX)

# TODO(avh): decide whether to use wgpu as default
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON
RELEASE_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=OFF
DEBUG_FLAGS = $(FLAGS) -DDEBUG:BOOL=ON
# EMSCRIPTEN_FLAGS = -DIMPLEMENTATION=emscripten -DCMAKE_TOOLCHAIN_FILE=../cmake/emscripten.cmake -DCMAKE_CXX_COMPILER=em++
EMSCRIPTEN_FLAGS = -DIMPLEMENTATION=emscripten -DCMAKE_CXX_COMPILER=em++
LOCAL_FLAGS = -DUSE_LOCAL_LIBS=ON 

demo: check-dependencies
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET_DEMO) && ./$(TARGET_DEMO)

# check for the existence of clang++ and cmake
check-dependencies:
	@command -v clang++ >/dev/null 2>&1 || { echo >&2 "Please install clang++ with 'sudo apt-get install clang' or 'brew install llvm'"; exit 1; }
	@command -v cmake >/dev/null 2>&1 || { echo >&2 "Please install cmake with 'sudo apt-get install cmake' or 'brew install cmake'"; exit 1; }

tests: check-dependencies
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)

libgpu: check-dependencies
	mkdir -p build && cd build && cmake .. $(RELEASE_FLAGS) && make -j$(NUM_JOBS) gpu

debug: check-dependencies
	mkdir -p build && cd build && cmake .. $(DEBUG_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

build: check-dependencies
	mkdir -p build && cd build && cmake .. $(RELEASE_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

emscripten: check-dependencies
	mkdir -p build && cd build && cmake .. $(EMSCRIPTEN_FLAGS) -DIMPLEMENTATION=emscripten && make -j$(NUM_JOBS) $(TARGET_ALL)

check-entr:
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }

watch-demo: check-entr check-dependencies
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_DEMO) && make -j$(NUM_JOBS) $(TARGET_DEMO) && ./$(TARGET_DEMO)"

watch-tests: check-entr check-dependencies
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_TESTS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)"

# experimental
watch-tests-wgpu: check-entr check-dependencies
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) $(USE_WGPU) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_TESTS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)"

watch-demo-local: check-entr check-dependencies
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) $(LOCAL_FLAGS) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_DEMO) && make -j$(NUM_JOBS) $(TARGET_DEMO) && ./$(TARGET_DEMO)"

watch-tests-local: check-entr check-dependencies
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) $(LOCAL_FLAGS) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_TESTS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)"

clean-build:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

clean:
	read -r -p "This will delete the contents of build/* and third_party/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/* third_party/fetchcontent/*
