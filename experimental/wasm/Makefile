FLAGS=--target=wasm32 -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--import-memory -Wl,--allow-undefined -fexceptions -std=c++17 -O3
# FLAGS=--target=wasm32-unknown-unknown -stdlib=libc++ -nostdlib++ -Wl,--no-entry -Wl,--export-all -Wl,--import-memory -Wl,--allow-undefined -fexceptions -std=c++17

.PHONY: all clean dump-obj dump-wasm dependencies server

all: build/hello.wasm dump-obj dump-wasm

watch:
	ls *.cpp *.h | entr make build/run.wasm

build/run.wasm: run.cpp Makefile
	clang++ $(FLAGS) -o build/run.wasm run.cpp
	
# cpp -> llvm ir
build/hello.ll: hello.cpp
	clang --target=wasm32 -emit-llvm -c -S hello.cpp -o build/hello.ll

# llvm ir -> wasm object file
build/hello.o: build/hello.ll
	llc -march=wasm32 -filetype=obj build/hello.ll -o build/hello.o

# Disassemble the WebAssembly object file
dump-obj:
	wasm-objdump -x build/hello.o

# Link the WebAssembly object file to a WebAssembly module
build/hello.wasm: build/hello.o
	wasm-ld \
		--no-entry \
		--export-all \
		-o build/hello.wasm \
		build/hello.o

dump-wasm:
	wasm-objdump -x build/hello.wasm

# TODO(avh): this is just a reminder note for now - remove it later
dependencies:
	brew install llvm
	brew install wabt

server: 
	python3 -m http.server 8000

clean:
	rm -f build/hello.ll build/hello.o build/hello.wasm build/run.wasm
