# Release 0.2.0 (draft)

Switched from
https://github.com/jspanchu/webgpu-dawn-binaries
to building from the dawn repository:
https://github.com/google/dawn

Commit hash:
5a00ab1fbc22d6ebbab39c901c1f90144e9b71e9 

Build with clang (assumes running from out/Release)

```
cmake -DBUILD_SHARED_LIBS=ON -DDAWN_BUILD_MONOLITHIC_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release ../..
```
Library artifact is at `src/dawn/native/libwebgpu_dawn.dylib` (7.4 MB)

# Release 0.1.0

https://github.com/jspanchu/webgpu-dawn-binaries
commit hash:
c0602d5d0466040f6e080d6cb7209860538f9f8d

Built with clang:
cmake .. -DCMAKE_C_COMPILER=clang DCMAKE_CXX_COMPILER=clang++
