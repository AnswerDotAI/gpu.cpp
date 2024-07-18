TARGET = shadertui

# Set up variables for cross-platform compatibility
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	MKDIR_CMD := if not exist build mkdir build
	RMDIR_CMD := rd /s /q
	SLASH := \\
	LDLIB_SUFFIX := dll
	EXPORT_CMD := set
else
	DETECTED_OS := $(shell uname)
	MKDIR_CMD := mkdir -p build
	RMDIR_CMD := rm -rf
	SLASH := /
	LDLIB_SUFFIX := so
	EXPORT_CMD := export
endif

# Determine the architecture
ifeq ($(DETECTED_OS), Windows_NT)
	ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		ARCH := x64
	else
		ARCH := x86
	endif
else
	ARCH := $(shell uname -m)
	ifeq ($(ARCH), x86_64)
		ARCH := x64
	else ifneq (,$(findstring arm, $(ARCH)))
		ARCH := arm
	endif
endif

# Paths
GPUCPP ?= $(shell pwd)

default: build_$(TARGET)_debug

run: build_$(TARGET)_debug dawnlib examples_$(TARGET)_build_$(TARGET)

build_$(TARGET)_release:
ifeq ($(OS),Windows_NT)
	@if not exist "build" $(MKDIR_CMD) && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release
else
	@command mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
endif

build_$(TARGET)_debug:
ifeq ($(OS),Windows_NT)
	@if not exist "build" $(MKDIR_CMD) && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build . --config Release
else
	@command mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build .
endif

examples_$(TARGET)_build_$(TARGET): check-clang dawnlib check-linux-vulkan
ifeq ($(DETECTED_OS), Windows_NT)
	cd examples$(SLASH)$(TARGET) && examples$(SLASH)$(TARGET)$(SLASH)build$(SLASH)$(TARGET).exe
else
	cd examples/$(TARGET) && $(MAKE) build/$(TARGET) && ./build/$(TARGET)/build
endif

dawnlib:
ifeq ($(DETECTED_OS), Windows_NT)
	@if not exist "$(GPUCPP)$(SLASH)third_party$(SLASH)lib$(SLASH)libdawn_$(ARCH)_$(BUILD_TYPE).dll" if not exist "$(GPUCPP)$(SLASH)third_party$(SLASH)lib$(SLASH)libdawn.dll" (make run_setup)
else
	@if [ ! -f "$(GPUCPP)/third_party/lib/libdawn_$(ARCH)_$(BUILD_TYPE).so" ] && [ ! -f "$(GPUCPP)/third_party/lib/libdawn.so" ] && [ ! -f "$(GPUCPP)/third_party/lib/libdawn_$(ARCH)_$(BUILD_TYPE).dylib" ]; then \
		$(MAKE) run_setup; \
	fi
endif

run_setup: check-python
ifeq ($(OS),Windows_NT)
	cd $(GPUCPP) && if exist "python3.exe" (python3 setup.py) else (if exist "python.exe" (python setup.py) else (echo "Python needs to be installed and in your path." & exit 1))
else
	cd $(GPUCPP) && (command -v python3 >/dev/null 2>&1 && python3 setup.py || python setup.py)
endif

clean:
ifeq ($(DETECTED_OS), Windows_NT)
	$(RMDIR_CMD) build
else
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build*
endif

check-python:
ifeq ($(OS),Windows_NT)
	@if not exist "$(shell where python3.exe 2>NUL)" if not exist "$(shell where python.exe 2>NUL)" (echo "Python needs to be installed and in your path." & exit 1)
else
	@command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { echo "Python needs to be installed and in your path."; exit 1; }
endif