GPUCPP=../..
FLAGS=-std=c++17 -s USE_WEBGPU=1 -s ASYNCIFY=1 -I$(GPUCPP)

.PHONY: default cmake check-emsdk browser clean

default: server

build/run.html: check-emsdk run.cpp custom_shell.html
	em++ run.cpp -o build/run.html \
		$(FLAGS) \
		--shell-file custom_shell.html \

build/run.wasm: check-emsdk run.cpp custom_shell.html
	em++ run.cpp -o build/run.wasm \
		$(FLAGS) \
		--shell-file custom_shell.html \

# make clean explicit here because custom_shell.html changes don't trigger a rebuild
cmake: check-emsdk clean
	emcmake cmake -B build && cmake --build build -j4
	python3 -m http.server --directory .

server: build/run.html
	@echo "\n┌───────────────────────────────────────────────────────────────────────────────────┐"
	@echo   "│  Open http://localhost:8000/build/run.html in your browser to see the output.     │"
	@echo	  "│                                                                                   │"
	@echo	  "│  Press Ctrl+C to stop the server.                                                 │"
	@echo   "└───────────────────────────────────────────────────────────────────────────────────┘\n\n"
	python3 -m http.server --directory .

open-browser:
	open http://127.0.0.1:8000/build/run.html

clean:
	rm -rf build/*

check-emsdk:
	@which em++ > /dev/null || (echo "emsdk not found. Please install emsdk and run 'source emsdk_env.sh' in the emsdk directory." && exit 1)
