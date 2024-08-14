GPUCPP=../../../
COMMON_FLAGS=-std=c++17 -s USE_WEBGPU=1 -s ASYNCIFY=1 -I$(GPUCPP)
# COMMON_FLAGS=-std=c++17 -s USE_WEBGPU=1 -I$(GPUCPP)
# Note - no spaces after comma
# enable exceptions to recover from WGSL failure
JS_FLAGS=-s EXPORTED_RUNTIME_METHODS=['UTF8ToString','setValue','addFunction','customPrint'] -s EXPORTED_FUNCTIONS=['_malloc','_free','_evaluate'] -s DISABLE_EXCEPTION_CATCHING=0 
WASM_FLAGS=-s STANDALONE_WASM -s ERROR_ON_UNDEFINED_SYMBOLS=0 -s EXPORTED_FUNCTIONS=['_evaluate'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] -DSTANDALONE_WASM
MODULARIZE_FLAGS=-s EXPORT_NAME='createModule' -s MODULARIZE=1 --bind
NO_MODULARIZE_FLAGS=-s EXPORTED_FUNCTIONS=['_evaluate'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] --bind

.PHONY: default cmake check-emsdk browser clean server debug

default: server

build/run.js: check-emsdk run.cpp
	em++ run.cpp -o build/run.js \
		$(COMMON_FLAGS) $(JS_FLAGS) $(MODULARIZE_FLAGS)

debug: check-emsdk run.cpp
	em++ -g -gsource-map run.cpp -o build/run.js \
		$(COMMON_FLAGS) $(JS_FLAGS) $(MODULARIZE_FLAGS)

build/run.wasm: check-emsdk run.cpp
	em++ run.cpp -o build/run.wasm \
		$(COMMON_FLAGS) $(WASM_FLAGS)

watch:
	@echo "Watching for changes..."
	ls run.cpp | entr -c make debug

server: build/run.js
	python3 run.py

clean:
	rm -rf build/*

check-emsdk:
	@which em++ > /dev/null || (echo "emsdk not found. Please install emsdk and run 'source emsdk_env.sh' in the emsdk directory." && exit 1)
