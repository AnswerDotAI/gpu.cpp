NUM_JOBS?=$(shell nproc)
CXX=clang++

TARGET = hello_gpu
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER="$(CXX)"
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON 

GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source

run:
	mkdir -p build && $(CXX) -v -std=c++17 -I$(GPUCPP) -I$(GPUCPP)/utils -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib run.cpp -ldl -ldawn -o ./build/$(TARGET) && ./build/$(TARGET)
	# mac
	# mkdir -p build && $(CXX) -std=c++17 -I$(GPUCPP) -I$(GPUCPP)/utils -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib -ldawn run.cpp -o ./build/$(TARGET) && $(LIBSPEC) && ./build/$(TARGET)

run-cmake:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)

watch:
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls $(GPUCPP)/utils/* | entr -s "rm -f $(TARGET) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)"

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
