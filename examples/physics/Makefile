NUM_JOBS ?= $(shell nproc)
CXX=clang++

TARGET = physics
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER="$(CXX)"
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON 
DEBUG_FLAGS = $(FLAGS) -DDEBUG:BOOL=ON

run:
	mkdir -p build && cd build && cmake .. $(FLAGS) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)

debug :
	mkdir -p build && cd build && cmake .. $(DEBUG_FLAGS) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)

watch:
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls ../* ../utils/* | entr -s "rm -f $(TARGET) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)"

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

