import os
import platform
import sys
import ssl
import urllib.request
from pathlib import Path

def get_os_name():
    system = platform.system()
    if system == "Windows":
        return "Windows 64-bit" if platform.machine().endswith('64') else "Windows 32-bit"
    elif system == "Darwin":
        return "macOS"
    elif system == "Linux":
        return "Linux"
    elif system == "FreeBSD":
        return "FreeBSD"
    elif system.startswith("CYGWIN"):
        return "Cygwin"
    else:
        return "Other"

def download_file(url, output_filename):
    total_downloaded = 0

    def report_progress(block_num, block_size, total_size):
        nonlocal total_downloaded
        total_downloaded += block_size
        print(f"\rDownloaded {total_downloaded // (1024 * 1024)} MB", end="")
    
    try:
        ssl._create_default_https_context = ssl._create_stdlib_context
        urllib.request.urlretrieve(url, output_filename, reporthook=report_progress)
        print(f"\nDownloaded {output_filename}")
        return True
    except Exception as e:
        print(f"\nFailed to download {output_filename}")
        print(f"Error: {str(e)}")
        return False

def check_os(os_name):
    print("\nChecking System")
    print("===============\n")
    print(f"  Operating System : {os_name}")
    supported = {"macOS", "Linux", "Windows 64-bit", "Windows 32-bit"}
    if os_name not in supported:
        print("Unsupported operating system")
        sys.exit(1)

def download_dawn(os_name, arch, build_type):
    print("\nDownload Dawn Library")
    print("=====================\n")

    lib_ext = {
        "macOS": "dylib",
        "Linux": "so",
        "Windows 64-bit": "dll",
        "Windows 32-bit": "dll",
    }

    outfile_map = {
        "macOS": f"third_party/lib/libdawn_{arch}_{build_type}.dylib",
        "Linux": f"third_party/lib/libdawn_{arch}_{build_type}.so",
        "Windows 64-bit": f"third_party\\lib\\libdawn_{arch}_{build_type}.dll",
        "Windows 32-bit": f"third_party\\lib\\libdawn_{arch}_{build_type}.dll",
    }
    fallback_map = {
        "macOS": "third_party/lib/libdawn.dylib",
        "Linux": "third_party/lib/libdawn.so",
        "Windows 64-bit": "third_party\\lib\\libdawn.dll",
        "Windows 32-bit": "third_party\\lib\\libdawn.dll",
    }

    url_map = {
        "macOS": f"https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn_{arch}_{build_type}.dylib",
        "Linux": f"https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn_{arch}_{build_type}.so",
        "Windows 64-bit": f"https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn_{arch}_{build_type}.dll",
        "Windows 32-bit": f"https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn_{arch}_{build_type}.dll",
    }
    fallback_url_map = {
        "macOS": "https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.dylib",
        "Linux": "https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.so",
        "Windows 64-bit": "https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn_x64.dll",
        "Windows 32-bit": "https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn_x86.dll",
    }

    outfile = Path(outfile_map.get(os_name))
    fallback_file = Path(fallback_map.get(os_name))
    url = url_map.get(os_name)
    fallback_url = fallback_url_map.get(os_name)

    cwd = Path.cwd()
    print(f"  Output File      : {outfile}")
    print(f"  Current Directory: {cwd}")
    print(f"  File Exists      : {cwd / outfile}")
    if outfile.exists():
        print(f"  File {outfile} already exists, skipping.")
        sys.exit(0)

    print(f"  URL              : {url}")
    print(f"  Download File    : {outfile}\n")
    print("  Downloading ...\n")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    if download_file(url, outfile):
        return

    print("\nPrimary file not found, attempting fallback download...\n")
    print(f"  Fallback URL     : {fallback_url}")
    print(f"  Fallback File    : {fallback_file}\n")

    if download_file(fallback_url, fallback_file):
        outfile.unlink(missing_ok=True)  # Remove partial download if needed
        fallback_file.rename(outfile)
        return

    print("Failed to download both primary and fallback files.")
    sys.exit(1)

def setup_env(os_name):
    print("\nEnvironment Setup")
    print("=================\n")
    
    current_dir = Path.cwd()
    lib_dir = current_dir / "third_party" / "lib"
    
    if os_name == "macOS":
        print("  Before running the program, run the following command or add it to your shell profile:")
        print(f"  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:{lib_dir}")
        
        with open("source", "w") as f:
            f.write(f"export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:{lib_dir}\n")
    if os_name == "Linux":
        print("  Before running the program, run the following command or add it to your shell profile:")
        print(f"  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{lib_dir}")
        
        with open("source", "w") as f:
            f.write(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{lib_dir}\n")
    if os_name.startswith("Windows"):
        print("  Before running the program, add the following path to your PATH environment variable:")
        print(f"  {lib_dir}")
        
        with open("source.bat", "w") as f:
            f.write(f"set PATH=%PATH%;{lib_dir}\n")

def main():
    os_name = get_os_name()
    arch = "x64" if platform.machine().endswith('64') else "x86"
    build_type = "Debug" if 'debug' in sys.argv else "Release"

    check_os(os_name)
    download_dawn(os_name, arch, build_type)
    setup_env(os_name)
    print()

if __name__ == "__main__":
    main()