CXX=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
FLAGS=-stdlib=libc++ -std=c++17 -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib run.cpp -ldl -ldawn
TARGET=hello_world

run: ./build/$(TARGET)
	$(LIBSPEC) && ./build/$(TARGET)

build/$(TARGET): run.cpp
	mkdir -p build && $(CXX) $(FLAGS) -DNDEBUG -o ./build/$(TARGET)

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
