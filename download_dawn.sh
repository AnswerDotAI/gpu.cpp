#!/bin/sh

lib_dir='./third_party/lib'

case "$(uname -s)" in
    Darwin)
        file="libdawn.dylib"
        url="https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.dylib"   
        ;;
    Linux)
        file="libdawn.so"
        url="https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.so"
        ;;
    *)
        echo "Not Supported"
        exit
        ;;
esac

mkdir -p $lib_dir
curl -L --progress-bar "$url" -o $lib_dir/$file

case "$os_name" in
    macOS)
        echo "  Before running the program, run the following command or add it to your shell profile:"
        echo "  export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$lib_dir"
        echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$lib_dir" > source
        ;;
    Linux)
        echo "  Before running the program, run the following command or add it to your shell profile:"
        echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$lib_dir"
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$lib_dir" > source
        ;;
esac
