
FLAGS=--target=wasm32 -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--import-memory -Wl,--allow-undefined -fexceptions -std=c++17 -O3


# Make sure you have:
# 1) emsdk installed
# 2) `source emsdk_env.sh` in the emsdk directory
build/run.wasm:
	em++ -sUSE_WEBGPU=1 -sUSE_GLFW=3 run.cpp -o build/run.wasm

cmake:
	emcmake cmake -B build && cmake --build build -j4
	npx http-server

browser:
	open http://127.0.0.1:8000/build/run.html

clean:
	rm -rf build/*
