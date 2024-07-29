.PHONY: all clean dump-obj dump-wasm dependencies server

all: run.wasm dump-obj dump-wasm

# Compile the C++ source file to LLVM IR
run.ll: run.cpp
	clang --target=wasm32 -emit-llvm -c -S run.cpp

# Assemble the LLVM IR to a WebAssembly object file
run.o: run.ll
	llc -march=wasm32 -filetype=obj run.ll

# Disassemble the WebAssembly object file
dump-obj:
	wasm-objdump -x run.o

# Link the WebAssembly object file to a WebAssembly module
# no entry point function
# export all functions
run.wasm: run.o
	wasm-ld \
		--no-entry \
		--export-all \
		-o run.wasm \
		run.o

dump-wasm:
	wasm-objdump -x run.wasm

# TODO(avh): this is just a reminder note for now - remove it later
dependencies:
	brew install llvm
	brew install wabt

server: 
	python3 -m http.server

clean:
	rm -f run.ll run.o run.wasm
