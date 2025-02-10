CXX=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
TARGET=float16
ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif
FLAGS=-std=c++17 $(STDLIB) -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib run.cpp -ldl -lwebgpu_dawn

run: ./build/$(TARGET) dawnlib
	$(LIBSPEC) && ./build/$(TARGET)

dawnlib: $(if $(wildcard $(GPUCPP)/third_party/lib/libwebgpu_dawn.so $(GPUCPP)/third_party/lib/libwebgpu_dawn.dylib),,run_setup)

run_setup: check-python
	cd $(GPUCPP) && python3 setup.py

build/$(TARGET): run.cpp
	mkdir -p build && $(CXX) $(FLAGS) -DNDEBUG -o ./build/$(TARGET)

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python needs to be installed and in your path."; exit 1; } 
