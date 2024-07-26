CXX=clang++
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source
NUM_JOBS?=$(shell nproc)
TARGET=microphone
ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif

PA_FLAG=-I$(GPUCPP)/third_party/headers/portaudio
PA_LIB=-lportaudio.2

FLAGS=-std=c++17 $(STDLIB) -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib $(PA_FLAG) run.cpp -ldl -ldawn $(PA_LIB)

run: ./build/$(TARGET) dawnlib
	$(LIBSPEC) && ./build/$(TARGET)

dawnlib: $(if $(wildcard $(GPUCPP)/third_party/lib/libdawn.so $(GPUCPP)/third_party/lib/libdawn.dylib),,run_setup)

run_setup: check-python
	cd $(GPUCPP) && python3 setup.py

build/$(TARGET): run.cpp check-portaudio
	mkdir -p build && $(CXX) $(FLAGS) -DNDEBUG -o ./build/$(TARGET)

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python needs to be installed and in your path."; exit 1; } 

check-portaudio:
	# check if portaudio.2.dylib or portaudio.so is in the third_party/lib directory
	@echo "Checking for portaudio library..."
	@if [ ! -f $(GPUCPP)/third_party/lib/libportaudio.2.dylib ] && [ ! -f $(GPUCPP)/third_party/lib/libportaudio.so ]; then \
		echo "Portaudio library not found. Please install portaudio and place the library in the third_party/lib directory."; \
		exit 1; \
	fi
	# check header file third_party/headers/portaudio/portaudio.h
	if [ ! -f $(GPUCPP)/third_party/headers/portaudio/portaudio.h ]; then \
		echo "Portaudio header file not found. Please install portaudio and place the header file in the third_party/headers/portaudio directory."; \
		exit 1; \
	fi
