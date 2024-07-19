# List of targets (folders in your examples directory)
TARGETS := gpu_puzzles hello_world matmul physics render shadertui

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
ifeq ($(OS),Windows_NT)
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

.PHONY: default all_release all_debug dawnlib run_setup check-python
.PHONY: $(addsuffix _release, $(TARGETS))
.PHONY: $(addsuffix _debug, $(TARGETS))
.PHONY: clean-all $(addprefix clean_, $(TARGETS))

default: all_debug

all_release: $(addsuffix _release, $(TARGETS))

all_debug: $(addsuffix _debug, $(TARGETS))

# Define Build Rules
define BUILD_RULES
$(1)_release:
ifeq ($(OS),Windows_NT)
	@if not exist "$(1)$(SLASH)build" $(MKDIR_CMD) "$(1)$(SLASH)build"
	cd $(1) && \
	cmake -B build -DCMAKE_BUILD_TYPE=Release && \
	cmake --build build --config Release
else
	$(MKDIR_CMD) $(1)$(SLASH)build
	cd $(1) && \
	cmake -B build -DCMAKE_BUILD_TYPE=Release && \
	cmake --build build --config Release
endif

$(1)_debug:
ifeq ($(OS),Windows_NT)
	@if not exist "$(1)$(SLASH)build" $(MKDIR_CMD) "$(1)$(SLASH)build"
	cd $(1) && \
	cmake -B build -DCMAKE_BUILD_TYPE=Debug && \
	cmake --build build --config Debug
else
	$(MKDIR_CMD) $(1)$(SLASH)build
	cd $(1) && \
	cmake -B build -DCMAKE_BUILD_TYPE=Debug && \
	cmake --build build --config Debug
endif
endef
# Apply Build Rules to each target in $(TARGETS)
$(foreach target,$(TARGETS),$(eval $(call BUILD_RULES,$(target))))

# Define Run Rules with Build if Executable Not Found
define RUN_RULES
run_$(1):
ifeq ($(OS),Windows_NT)
	@if exist $(1)$(SLASH)build$(SLASH)Release$(SLASH)$(1).exe ( \
		echo "Running $(1)" && \
		$(1)$(SLASH)build$(SLASH)Release$(SLASH)$(1).exe \
	) else ( \
		echo "Executable not found; building $(1) first" && \
		$(MAKE) $(1)_release && \
		$(1)$(SLASH)build$(SLASH)Release$(SLASH)$(1).exe \
	)
else
	@if [ -x "$(shell find $(1)/build -type f -executable -name $(1) | head -n 1)" ]; then \
		exe_path=$(shell find $(1)/build -type f -executable -name $(1) | head -n 1) && \
		echo "Running $(1)" && \
		$$exe_path; \
	else \
		echo "Executable not found; building $(1) first" && \
		cd $(1) && $(MAKE) $(1)_release && \
		exe_path=$(shell find $(1)/build -type f -executable -name $(1) | head -n 1) && \
		$$exe_path; \
	fi
endif
endef
# Apply Run Rules to each target in $(TARGETS)
$(foreach target,$(TARGETS),$(eval $(call RUN_RULES,$(target))))

# Clean rules for cleaning specific targets
define CLEAN_RULES
clean_$(1):
ifeq ($(OS),Windows_NT)
	if exist $(1)$(SLASH)build ( $(RMDIR_CMD) $(1)$(SLASH)build )
else
	find $(1) -name build -type d | xargs rm -rf
endif
endef
$(foreach target,$(TARGETS),$(eval $(call CLEAN_RULES,$(target))))

clean-all: $(addprefix clean_, $(TARGETS))

dawnlib:
ifeq ($(OS),Windows_NT)
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

check-python:
ifeq ($(OS),Windows_NT)
	@if not exist "$(shell where python3.exe 2>NUL)" if not exist "$(shell where python.exe 2>NUL)" (echo "Python needs to be installed and in your path." & exit 1)
else
	@command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { echo "Python needs to be installed and in your path."; exit 1; }
endif