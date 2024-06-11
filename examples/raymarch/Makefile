NUM_JOBS ?= $(shell nproc)
CXX=clang++

TARGET = raymarch
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER="$(CXX)"
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON 

run:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)

watch:
	mkdir -p build && cd build && cmake .. $(FASTBUILD_FLAGS) && ls ../* | entr -s "rm -f $(TARGET) && make -j$(NUM_JOBS) $(TARGET) && ./$(TARGET)"


clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*
