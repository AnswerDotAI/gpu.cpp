NUM_JOBS=$(shell nproc)
CXX=clang++
TARGET_DEMO=run_demo
TARGET_TESTS=run_tests
USE_LOCAL=-DUSE_LOCAL_LIBS=ON

.PHONY: demo tests libgpu build-debug build check-entr watch-demo watch-tests clean

FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX)
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON 
RELEASE_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=OFF
DEBUG_FLAGS = $(FLAGS) -DDEBUG:BOOL=ON
# EMSCRIPTEN_FLAGS = -DIMPLEMENTATION=emscripten -DCMAKE_TOOLCHAIN_FILE=../cmake/emscripten.cmake -DCMAKE_CXX_COMPILER=em++
EMSCRIPTEN_FLAGS = -DIMPLEMENTATION=emscripten -DCMAKE_CXX_COMPILER=em++

demo:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET_DEMO) && ./$(TARGET_DEMO)

tests:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)

libgpu:
	mkdir -p build && cd build && cmake .. $(RELEASE_FLAGS) && make -j$(NUM_JOBS) gpu

build-debug:
	mkdir -p build && cd build && cmake .. $(DEBUG_FLAGS) && make -j$(NUM_JOBS)

build:
	mkdir -p build && cd build && cmake .. $(RELEASE_FLAGS) && make -j$(NUM_JOBS)

check-entr:
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }

watch-demo: check-entr
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_DEMO) && make -j$(NUM_JOBS) $(TARGET_DEMO) && ./$(TARGET_DEMO)"

watch-tests:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls ../* ../utils/* | entr -s "rm -f $(TARGET_TESTS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)"

# TODO(avh): fix
watch-tests-local:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) -DUSE_LOCAL_LIBS=ON && ls ../* ../utils/* | entr -s "rm -f $(TARGET_TESTS) && make -j$(NUM_JOBS) $(TARGET_TESTS) && ./$(TARGET_TESTS)"

emscripten:
	mkdir -p build && cd build && cmake .. $(EMSCRIPTEN_FLAGS) -DIMPLEMENTATION=emscripten && make -j$(NUM_JOBS)

clean-build:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

clean:
	read -r -p "This will delete the contents of build/* and third_party/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/* third_party/fetchcontent/*


