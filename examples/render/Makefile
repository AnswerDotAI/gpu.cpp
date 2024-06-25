CXX=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
TARGET=render

run: ./build/$(TARGET)
	$(LIBSPEC) && ./build/$(TARGET)

build/$(TARGET): run.cpp
	mkdir -p build && $(CXX) -v -std=c++17 -I$(GPUCPP) -I$(GPUCPP)/utils -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib run.cpp -ldl -ldawn -o ./build/$(TARGET)

watch:
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }
	mkdir -p build && ls | entr -s "rm -f ./build/$(TARGET) && make -j$(NUM_JOBS) ./build/$(TARGET) && ./build/$(TARGET)"

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
