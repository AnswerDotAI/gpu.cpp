# List of targets (folders in your examples directory)
TARGETS := float16 gpu_puzzles hello_world matmul physics render shadertui transpose

GPUCPP ?= $(shell pwd)/..
CXX=clang++
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
BUILD ?= debug

ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif

FLAGS=-std=c++17 $(STDLIB) -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib
LFLAGS=-ldl -lwebgpu_dawn

.PHONY: default all_release all_debug dawnlib run_setup check-python
.PHONY: $(addsuffix _release, $(TARGETS))
.PHONY: $(addsuffix _debug, $(TARGETS))
.PHONY: clean-all $(addprefix clean_, $(TARGETS))

all: all_$(BUILD)

all_release: $(addsuffix _release, $(TARGETS))

all_debug: $(addsuffix _debug, $(TARGETS))

define BUILD_RULES
$(1)_release: dawnlib
	mkdir -p $(1)/build
	$(CXX) $(FLAGS) $(1)/run.cpp $(LFLAGS) -DNDEBUG -o $(1)/build/$(1)

$(1)_debug: dawnlib
	mkdir -p $(1)/build
	$(CXX) $(FLAGS) $(1)/run.cpp $(LFLAGS) -o $(1)/build/$(1)

$(1): $(1)_$(BUILD)
endef

define RUN_RULES
run_$(1): $(1)
	# note directory context is important in the case of shadertui which loads the shader from disk
	$(LIBSPEC) && cd $(1) && ./build/$(1)
endef

# Clean rules for cleaning specific targets
define CLEAN_RULES
clean_$(1):
	rm -rf $(1)/build
endef

$(foreach target,$(TARGETS),$(eval $(call BUILD_RULES,$(target))))
$(foreach target,$(TARGETS),$(eval $(call RUN_RULES,$(target))))
$(foreach target,$(TARGETS),$(eval $(call CLEAN_RULES,$(target))))

clean: $(addprefix clean_, $(TARGETS))

dawnlib: $(if $(wildcard $(GPUCPP)/third_party/lib/libdawn.so $(GPUCPP)/third_party/lib/libdawn.dylib),,run_setup)

run_setup: check-python
	cd $(GPUCPP) && (command -v python3 >/dev/null 2>&1 && python3 setup.py || python setup.py)

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python needs to be installed and in your path."; exit 1; } 
