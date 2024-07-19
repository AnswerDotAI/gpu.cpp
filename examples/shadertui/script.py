import shutil
import os

# List of shader files to copy
shader_files = [
    "default.wgsl",
    "roundrect1.wgsl",
    "roundrect2.wgsl",
    "shapes.wgsl",
    "boat.wgsl",
    "gradient_flow.wgsl",
    "wave_interference.wgsl",
    "reaction_diffusion.wgsl",
    "voronoi.wgsl",
    "fluid.wgsl",
    "aurora.wgsl",
    "julia.wgsl",
    "mandelbrot.wgsl",
    "particles.wgsl",
    "default.wgsl",
]  # Add more file names as needed


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except:
        print("Unexpected error:", sys.exc_info())


if __name__ == "__main__":
    # clear screen
    os.system("cls" if os.name == "nt" else "clear")
    print(
        "\nThis script is meant to run alongside the shadertui runner. To start shadertui open a separate terminal and run `make` from this directory.\n"
    )
    input("Press return/enter to continue...")

    for shader in shader_files:
        if os.path.exists(shader):
            copy_file(shader, "shader.wgsl")
            input("Press return/enter to continue...")
        else:
            print(f"File {shader} does not exist.")
            break

    print("All files processed.")
