.PHONY: all build dawn run test

all: build

build:
	cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
	cmake --build build

dawn:
	tools/build_dawn.sh

run: build
	./build/hello_gpu

test:
	./test
