CXX=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
CODEPATH = find . ../../utils ../../ -maxdepth 1 -type f
TARGET=matmul
ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif
FLAGS=-std=c++17 $(STDLIB) -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib run.cpp -ldl -lwebgpu_dawn

run: ./build/$(TARGET)
	$(LIBSPEC) && ./build/$(TARGET)

run_with_metal_profiler: ./build/$(TARGET)_with_metal_profiler
	$(LIBSPEC) && export METAL_CAPTURE_ENABLED=1 && ./build/$(TARGET)_with_metal_profiler

run_with_time_profiler: ./build/$(TARGET)_with_metal_profiler
	$(LIBSPEC) && xcrun xctrace record --template 'Time Profiler' --launch -- ./build/$(TARGET)_with_metal_profiler

# Use clang -v to see the include paths
# Note in this example optimization is turned on
build/$(TARGET): run.cpp
	mkdir -p build && $(CXX) $(FLAGS) -o ./build/$(TARGET)

build/$(TARGET)_with_metal_profiler: run.cpp
	mkdir -p build && $(CXX) $(FLAGS) -o ./build/$(TARGET)_with_metal_profiler $(GPUCPP)/experimental/profiler/metal.mm -framework metal -framework Foundation -DMETAL_PROFILER -g
	install_name_tool -change @rpath/libwebgpu_dawn.dylib $(GPUCPP)/third_party/lib/libwebgpu_dawn.dylib ./build/$(TARGET)_with_metal_profiler

watch: 
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }
	mkdir -p build && $(CODEPATH) | entr -s "$(LIBSPEC) && rm -f ./build/$(TARGET) && make -j$(NUM_JOBS) ./build/$(TARGET) && ./build/$(TARGET)"

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
