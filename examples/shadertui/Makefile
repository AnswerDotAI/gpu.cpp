CXX=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
TARGET=shadertui
CODEPATH = find . ../../utils ../../ -maxdepth 1 -type f
ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif
FLAGS=-std=c++17 $(STDLIB) -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib run.cpp -ldl -lwebgpu_dawn


run: ./build/$(TARGET)
	$(LIBSPEC) && ./build/$(TARGET)

# Use clang -v to see the include paths
build/$(TARGET): run.cpp
	mkdir -p build && $(CXX) $(FLAGS) -o ./build/$(TARGET)

watch: 
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }
	mkdir -p build && $(CODEPATH) | entr -s "$(LIBSPEC) && rm -f ./build/$(TARGET) && make -j$(NUM_JOBS) run"

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
