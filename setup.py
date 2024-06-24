import os
import platform
import sys
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
    supported = {"macOS", "Linux"}
    if os_name not in supported:
        print("Unsupported operating system")
        sys.exit(1)

def download_dawn(os_name):
    print("\nDownload Dawn Library")
    print("=====================\n")

    outfile_map = {
        "macOS": "third_party/lib/libdawn.dylib",
        "Linux": "third_party/lib/libdawn.so",
    }
    url_map = {
        "macOS": "https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.dylib",
        "Linux": "https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.so",
    }

    outfile = outfile_map.get(os_name)
    url = url_map.get(os_name)

    if not outfile or not url:
        print(f"No download information for {os_name}")
        return

    print(f"  URL              : {url}")
    print(f"  Download File    : {outfile}\n")
    print("  Downloading ...\n")

    if Path(outfile).exists():
        print(f"  File {outfile} already exists, skipping.")
        return

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    download_file(url, outfile)

def setup_env(os_name):
    print("\nEnvironment Setup")
    print("=================\n")
    
    current_dir = os.getcwd()
    lib_dir = os.path.join(current_dir, "third_party", "lib")
    
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

def main():
    os_name = get_os_name()
    check_os(os_name)
    download_dawn(os_name)
    setup_env(os_name)
    print()

if __name__ == "__main__":
    main()
