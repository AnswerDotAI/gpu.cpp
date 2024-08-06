GPUCPP=../..
COMMON_FLAGS=-std=c++17 -s USE_WEBGPU=1 -s ASYNCIFY=1 -I$(GPUCPP)
# Note - no spaces after comma
# enable exceptions to recover from WGSL failure
JS_FLAGS=-s EXPORTED_RUNTIME_METHODS=['UTF8ToString','setValue','addFunction'] -s EXPORTED_FUNCTIONS=['_malloc','_free','_executeKernel'] -s NO_DISABLE_EXCEPTION_CATCHING 
WASM_FLAGS=-s STANDALONE_WASM -s ERROR_ON_UNDEFINED_SYMBOLS=0 -s EXPORTED_FUNCTIONS=['_executeKernel'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] -DSTANDALONE_WASM
MODULARIZE_FLAGS=-s EXPORT_NAME='createModule' -s MODULARIZE=1 --bind
NO_MODULARIZE_FLAGS=-s EXPORTED_FUNCTIONS=['_executeKernel'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] --bind

.PHONY: default cmake check-emsdk browser clean server

default: server

# build/run.html: check-emsdk run.cpp
#	em++ run.cpp -o build/run.html \
#		$(COMMON_FLAGS) $(JS_FLAGS) --shell-file ./custom_shell.html

build/run.js: check-emsdk run.cpp
	em++ run.cpp -o build/run.js \
		$(COMMON_FLAGS) $(JS_FLAGS) $(MODULARIZE_FLAGS)

build/run.wasm: check-emsdk run.cpp
	em++ run.cpp -o build/run.wasm \
		$(COMMON_FLAGS) $(WASM_FLAGS)

server: build/run.js
	python3 run.py

clean:
	rm -rf build/*

check-emsdk:
	@which em++ > /dev/null || (echo "emsdk not found. Please install emsdk and run 'source emsdk_env.sh' in the emsdk directory." && exit 1)
