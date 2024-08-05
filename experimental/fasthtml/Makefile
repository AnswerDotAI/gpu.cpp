GPUCPP=../..
FLAGS=-std=c++17 -s USE_WEBGPU=1 -s ASYNCIFY=1 -I$(GPUCPP)

.PHONY: default cmake check-emsdk browser clean

default: server

build/run.html: check-emsdk run.cpp
	em++ run.cpp -o build/run.html \
		$(FLAGS) --shell-file ./custom_shell.html

build/run.js: check-emsdk run.cpp
	em++ run.cpp -o build/run.js --shell-file ./custom_shell.html \
		$(FLAGS)

build/run.wasm: check-emsdk run.cpp
	em++ run.cpp -o build/run.wasm \
		$(FLAGS)

server: build/run.wasm
	python3 run.py

clean:
	rm -rf build/*

check-emsdk:
	@which em++ > /dev/null || (echo "emsdk not found. Please install emsdk and run 'source emsdk_env.sh' in the emsdk directory." && exit 1)
