run:
	mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DWEBGPU_BACKEND=DAWN -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && make wgpu_tutorial && ./wgpu_tutorial

watch:
	mkdir -p build && cd build && ls ../* | entr -s "cmake .. -DCMAKE_BUILD_TYPE=Debug -DWEBGPU_BACKEND=DAWN -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && make wgpu_tutorial && ./wgpu_tutorial"

clean:
	read -r -p "Are you sure? [CTRL-C to abort] " response && rm -rf build/*
