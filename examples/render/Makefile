NUM_JOBS ?= $(shell nproc)
CXX ?= clang++

TARGET = render
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER="$(CXX)"
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON 

GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= export DYLD_LIBRARY_PATH=$(LIBDIR)

run:
	mkdir -p build && $(CXX) -std=c++17 -I$(GPUCPP) -I$(GPUCPP)/utils -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib -ldawn run.cpp -o ./build/$(TARGET) && $(LIBSPEC) && ./build/$(TARGET)

run-cmake:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)

watch:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls ../* | entr -s "rm -f $(TARGET) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)"


clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
