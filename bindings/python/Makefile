CXX=clang++
PYTHON=python3
GPUCPP ?= $(PWD)/../..
LIBDIR ?= $(GPUCPP)/third_party/lib
LIBSPEC ?= . $(GPUCPP)/source

ifeq ($(shell $(CXX) -std=c++17 -x c++ -E -include array - < /dev/null > /dev/null 2>&1 ; echo $$?),0)
    STDLIB :=
else
    STDLIB := -stdlib=libc++
endif

FLAGS=-shared -fPIC -std=c++17 $(STDLIB) -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib -ldawn \
  `python3 -m pybind11 --includes` \
  `python3-config --include --ldflags --embed`

SUFFIX=$(shell $(PYTHON)-config --extension-suffix)

gpu_cpp$(SUFFIX): gpu_cpp.cpp 
	$(CXX) $(FLAGS) -o $@ $<
	install_name_tool -change @rpath/libdawn.dylib $(LIBDIR)/libdawn.dylib gpu_cpp$(SUFFIX)

test: test_gpu_cpp.py gpu_cpp$(SUFFIX)
	$(PYTHON) test_gpu_cpp.py

.PHONY: test
