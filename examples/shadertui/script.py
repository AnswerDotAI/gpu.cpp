from pathlib import Path
from shutil import copyfile

shaders = ('default.wgsl roundrect1.wgsl roundrect2.wgsl shapes.wgsl boat.wgsl gradient_flow.wgsl wave_interference.wgsl '
    'reaction_diffusion.wgsl voronoi.wgsl fluid.wgsl aurora.wgsl julia.wgsl mandelbrot.wgsl particles.wgsl default.wgsl').split()


def main():
    root = Path(__file__).parent
    print(f'Run `cd {root} && ../../build/shadertui_gpu` in another terminal.')
    input('Press return to begin...')
    for shader in shaders:
        copyfile(root / shader, root / 'shader.wgsl')
        input(f'Loaded {shader}. Press return to continue...')


if __name__ == '__main__': main()
