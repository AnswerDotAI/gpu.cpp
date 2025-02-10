# Dictionary of header files and their relative paths
header_files = {
    "#include \"webgpu/webgpu.h\"": "third_party/headers/webgpu/webgpu.h",
    "#include \"numeric_types/half.hpp\"": "numeric_types/half.hpp",
    "#include \"utils/logging.hpp\"": "utils/logging.hpp"
}

def main():
    # File paths
    source_file_path = "gpu.hpp"
    output_file_path = "build/gpu.hpp"

    # Open source file and read contents
    with open(source_file_path, "r") as source:
        file_contents = source.read()

    # Ergodic over header files
    for key, value in header_files.items():

        # Replace header files
        with open(value, "r") as header_file:
            header_file_contents = header_file.read()
        file_contents = file_contents.replace(key, header_file_contents)
        

    # Open output file
    with open(output_file_path, "w") as output:
        # Write contents to output file
        output.write(file_contents)

if __name__ == "__main__":
    main()