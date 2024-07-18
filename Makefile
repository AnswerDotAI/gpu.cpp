NUM_JOBS=$(shell nproc 2>/dev/null || echo 1)
CXX=clang++

.PHONY: default examples/hello_world/build/hello_world tests libgpu debug build check-entr check-clang clean-build clean clean-dawnlib all watch-tests docs

# Set up variables for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    MKDIR_CMD := if not exist build mkdir build
    RMDIR_CMD := rmdir
    SLASH := \\
    LS_CMD := dir
    LDLIB_SUFFIX := dll
    EXPORT_CMD := set
else
    DETECTED_OS := $(shell uname)
    MKDIR_CMD := mkdir -p build
    RMDIR_CMD := rm -rf
    SLASH := /
    LS_CMD := ls
    LDLIB_SUFFIX := so
    EXPORT_CMD := export
endif

# Determine the architecture
ifeq ($(DETECTED_OS), Windows)
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

# Determine the build type
BUILD_TYPE ?= Release
LOWER_BUILD_TYPE ?= $(shell python3 -c "print('$(BUILD_TYPE)'.lower())")

# Paths
GPUCPP ?= $(shell pwd)
LIBDIR ?= $(GPUCPP)$(SLASH)third_party$(SLASH)lib
LIBSPEC ?= . $(GPUCPP)$(SLASH)source

default: examples_hello_world_build_hello_world

examples_hello_world_build_hello_world: check-clang dawnlib examples/hello_world/run.cpp check-linux-vulkan
ifeq ($(DETECTED_OS), Windows)
	cd examples$(SLASH)hello_world && $(MAKE) build_hello_world_$(LOWER_BUILD_TYPE)
else
	$(LIBSPEC) && cd examples$(SLASH)hello_world && $(MAKE) build_hello_world_$(LOWER_BUILD_TYPE)
endif

# We use the custom "shell" based condition to check files cross-platform
dawnlib: 
ifeq ($(DETECTED_OS), Windows)
	@if not exist "$(LIBDIR)$(SLASH)libdawn_$(ARCH)_$(BUILD_TYPE).dll" if not exist "$(LIBDIR)$(SLASH)libdawn.dll" $(MAKE) run_setup
else
	@if [ ! -f "$(LIBDIR)$(SLASH)libdawn_$(ARCH)_$(BUILD_TYPE).so" ] && [ ! -f "$(LIBDIR)$(SLASH)libdawn.so" ] && [ ! -f "$(LIBDIR)$(SLASH)libdawn_$(ARCH)_$(BUILD_TYPE).dylib" ]; then \
        $(MAKE) run_setup; \
    fi
endif

run_setup: check-python
ifeq ($(DETECTED_OS), Windows)
	python3 setup.py
else
	python3 >/dev/null 2>&1 && python3 setup.py
endif

all: dawnlib check-clang check-linux-vulkan
	cd examples$(SLASH)gpu_puzzles && make build$(SLASH)gpu_puzzles
	cd examples$(SLASH)hello_world && make build$(SLASH)hello_world
	cd examples$(SLASH)matmul && make build$(SLASH)mm
	cd examples$(SLASH)physics && make build$(SLASH)physics
	cd examples$(SLASH)render && make build$(SLASH)render

docs: Doxyfile
	doxygen Doxyfile

################################################################################
# cmake targets (optional - precompiled binaries is preferred)
################################################################################

CMAKE_CMD = $(MKDIR_CMD) && cd build && cmake ..
# Add --trace to see the cmake commands
FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX) -DABSL_INTERNAL_AT_LEAST_CXX20=OFF
FASTBUILD_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=ON
DEBUG_FLAGS = $(FLAGS) -DDEBUG:BOOL=ON
RELEASE_FLAGS = $(FLAGS) -DFASTBUILD:BOOL=OFF
TARGET_LIB=gpu

libgpu-cmake: check-clang check-cmake
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) gpu

debug-cmake: check-clang check-cmake
	$(CMAKE_CMD) $(DEBUG_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

all-cmake: check-clang check-cmake
	$(CMAKE_CMD) $(RELEASE_FLAGS) && make -j$(NUM_JOBS) $(TARGET_ALL)

################################################################################
# Cleanup
################################################################################

clean-dawnlib:
	$(RMDIR_CMD) $(LIBDIR)$(SLASH)libdawn*.*

clean:
ifeq ($(DETECTED_OS), Windows)
	@if exist build $(RMDIR_CMD) build /s /q
	@if exist examples$(SLASH)gpu_puzzles$(SLASH)build $(RMDIR_CMD) examples$(SLASH)gpu_puzzles$(SLASH)build /s /q
	@if exist examples$(SLASH)hello_world$(SLASH)build $(RMDIR_CMD) examples$(SLASH)hello_world$(SLASH)build /s /q
	@if exist examples$(SLASH)matmul$(SLASH)build$(SLASH)mm $(RMDIR_CMD) examples$(SLASH)matmul$(SLASH)build$(SLASH)mm /s /q
	@if exist examples$(SLASH)physics$(SLASH)build $(RMDIR_CMD) examples$(SLASH)physics$(SLASH)build /s /q
	@if exist examples$(SLASH)render$(SLASH)build $(RMDIR_CMD) examples$(SLASH)render$(SLASH)build /s /q
	$(MKDIR_CMD)
else
	@command read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build*
	rm -rf examples/gpu_puzzles/build*
	rm -rf examples/hello_world/build*
	rm -rf examples/matmul/build/mm
	rm -rf examples/physics/build*
	rm -rf examples/render/build*
endif

clean-all:
ifeq ($(DETECTED_OS), Windows)
	@if exist build $(RMDIR_CMD) build /s /q
	$(RMDIR_CMD) third_party$(SLASH)fetchcontent /s /q
	$(RMDIR_CMD) third_party$(SLASH)gpu-build /s /q
	$(RMDIR_CMD) third_party$(SLASH)gpu-subbuild /s /q
	$(RMDIR_CMD) third_party$(SLASH)gpu-src /s /q
	$(RMDIR_CMD) third_party$(SLASH)lib /s /q
	$(MKDIR_CMD)
else
	read -r -p "This will delete the contents of build/* and third_party/*. Are you sure? [CTRL-C to abort] " response && rm -rf build* third_party/fetchcontent* third_party/gpu-build third_party/gpu-subbuild third_party/gpu-src third_party/lib/libdawn* third_party/lib/libdawn_$(ARCH)_$(BUILD_TYPE).*
endif


################################################################################
# Checks
################################################################################

# check for the existence of clang++
check-clang:
ifeq ($(OS),Windows_NT)
	@if not exist "$(shell where clang++.exe 2>NUL)" (echo "Please install clang++ with 'sudo apt-get install clang' or 'brew install llvm'" & exit 1)
else
	@command -v clang++ >/dev/null 2>&1 || { echo >&2 "Please install clang++ with 'sudo apt-get install clang' or 'brew install llvm'"; exit 1; }
endif

# check for the existence of entr
check-entr:
ifeq ($(OS),Windows_NT)
	@if not exist "$(shell where entr.exe 2>NUL)" (echo "Please install entr with 'brew install entr' or 'sudo apt-get install entr'" & exit 1)
else
	@command -v entr >/dev/null 2>&1 || { echo >&2 "Please install entr with 'brew install entr' or 'sudo apt-get install entr'"; exit 1; }
endif

# check for the existence of cmake
check-cmake:
ifeq ($(OS),Windows_NT)
	@if not exist "$(shell where cmake.exe 2>NUL)" (echo "Please install cmake with 'sudo apt-get install cmake' or 'brew install cmake'" & exit 1)
else
	@command -v cmake >/dev/null 2>&1 || { echo >&2 "Please install cmake with 'sudo apt-get install cmake' or 'brew install cmake'"; exit 1; }
endif

# check for the existence of python3
check-python:
ifeq ($(OS),Windows_NT)
	@if not exist "$(shell where python3.exe 2>NUL)" (echo "Python needs to be installed and in your path." & exit 1)
else
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python needs to be installed and in your path."; exit 1; }
endif

# check the existence of Vulkan (Linux only)
check-linux-vulkan:
	@echo "Checking system type and Vulkan availability..."
ifeq ($(OS),Linux)
	@command -v vulkaninfo >/dev/null 2>&1 && { echo "Vulkan is installed."; vulkaninfo; } || { echo "Vulkan is not installed. Please install Vulkan drivers to continue. On Debian / Ubuntu: sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools"; exit 1; }
else
	@echo "Non-Linux system detected. Skipping Vulkan check.";
endif