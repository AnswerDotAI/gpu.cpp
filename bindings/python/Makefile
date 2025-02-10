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

FLAGS=-shared -fPIC -std=c++17 $(STDLIB) -I$(GPUCPP) -I$(GPUCPP)/third_party/headers -L$(GPUCPP)/third_party/lib -lwebgpu_dawn \
  `python3 -m pybind11 --includes` \
  `python3-config --includes --ldflags`

SUFFIX=$(shell $(PYTHON)-config --extension-suffix)

gpu_cpp$(SUFFIX): gpu_cpp.cpp 
	$(CXX) $(FLAGS) -o $@ $<

test: test_gpu_cpp.py gpu_cpp$(SUFFIX)
	$(PYTHON) test_gpu_cpp.py

.PHONY: test
