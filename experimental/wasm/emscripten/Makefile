FLAGS=--target=wasm32 -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--import-memory -Wl,--allow-undefined -fexceptions -std=c++17 -O3

# For all targets, make sure emsdk is installed and `source emsdk_env.sh` is run in the emsdk directory


.PHONY: cmake check-emsdk browser clean

# make clean explicit here because custom_shell.html changes don't trigger a rebuild
cmake: check-emsdk clean
	emcmake cmake -B build && cmake --build build -j4
	npx http-server

build/run.wasm: check-emsdk run.cpp
	em++ -sUSE_WEBGPU=1 -sUSE_GLFW=3 run.cpp -o build/run.wasm

browser:
	open http://127.0.0.1:8000/build/run.html

clean:
	rm -rf build/*

check-emsdk:
	@which emcc > /dev/null || (echo "emsdk not found. Please install emsdk and run 'source emsdk_env.sh' in the emsdk directory." && exit 1)
